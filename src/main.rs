use std::any::Any;
use std::future::Future;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use datafusion::arrow::array::{
    Array, BinaryArray, BinaryBuilder, BooleanArray, LargeStringBuilder, RecordBatch, UInt64Builder,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::error::ArrowError;
use datafusion::arrow::util::pretty::print_batches;
use datafusion::common::{exec_err, Result as DataFusionResult, ScalarValue};
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::functions_aggregate::count::count;
use datafusion::logical_expr::utils::COUNT_STAR_EXPANSION;
use datafusion::logical_expr::{ColumnarValue, ScalarUDF, ScalarUDFImpl, Signature, Volatility};
use datafusion::prelude::*;
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng};

#[tokio::main]
async fn main() {
    run().await.unwrap();
}

const SEARCH_TERM: &'static str = "host.id";
// const SEARCH_TERM: &'static str = "row12";

#[rustfmt::skip]
const MODES: &[Mode] = &[
    Mode::FilterJson,
    Mode::FilterBatson,
];

const ROWS: usize = 1_000_000;
const COMPRESSION: &str = "zstd(1)";
const PUSHDOWN_FILTERS: bool = false;

async fn run() -> DataFusionResult<()> {
    let config = SessionConfig::new()
        .set_str("datafusion.sql_parser.dialect", "postgres")
        .set_str("datafusion.execution.parquet.compression", COMPRESSION)
        .set_bool("datafusion.execution.parquet.pushdown_filters", PUSHDOWN_FILTERS)
        // .set_usize("datafusion.execution.batch_size", 128)
        ;
    let ctx = SessionContext::new_with_config(config);

    create_or_load(&ctx, "records", create_table).await?;

    println!("running queries...");
    for mode in MODES {
        let start = Instant::now();
        let df = build_dataframe(&ctx, mode).await?;

        let df = df.aggregate(vec![], vec![count(Expr::Literal(COUNT_STAR_EXPANSION))])?;
        // let df = df.explain(false, true)?;

        let batches = df.collect().await?;
        let elapsed = start.elapsed();
        print_batches(&batches)?;
        println!("mode: {mode:?} {COMPRESSION}, pushdown_filters: {PUSHDOWN_FILTERS}, query took {elapsed:?}");
    }

    // datafusion_functions_json::register_all(&mut ctx)?;
    // let batson_contains = ScalarUDF::from(BatsonContains::new());
    // ctx.register_udf(batson_contains);
    //
    // let df = ctx.sql(r#"
    //     select count(*) from records where json_contains(json, 'host.id')
    // "#).await?;
    //
    // let plan = df.clone().create_physical_plan().await?;
    // dbg!(&plan);
    //
    // let batches = df.collect().await?;
    // print_batches(&batches)?;

    Ok(())
}

#[allow(dead_code)]
#[derive(Debug, Copy, Clone)]
enum Mode {
    FilterJson,
    FilterBatson,
}

async fn build_dataframe(ctx: &SessionContext, mode: &Mode) -> DataFusionResult<DataFrame> {
    let batson_contains = ScalarUDF::from(BatsonContains::new());

    let json_contains = datafusion_functions_json::udfs::json_contains_udf();
    let df = ctx.table("records").await?;

    match mode {
        Mode::FilterJson => df.filter(json_contains.call(vec![col("json"), lit(SEARCH_TERM)])),
        Mode::FilterBatson => {
            df.filter(batson_contains.call(vec![col("batson"), lit(SEARCH_TERM)]))
        }
    }
}

#[derive(Debug)]
struct BatsonContains {
    signature: Signature,
}

impl BatsonContains {
    fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Binary, DataType::Utf8],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for BatsonContains {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "batson_contains"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> DataFusionResult<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke(&self, args: &[ColumnarValue]) -> DataFusionResult<ColumnarValue> {
        let Some(ColumnarValue::Array(batson_column)) = args.first() else {
            return exec_err!(
                "batson_contains expects 2 arguments (batson_data, key), got no arguments"
            );
        };
        let Some(batson_column) = batson_column.as_any().downcast_ref::<BinaryArray>() else {
            return exec_err!(
                "batson_contains expects 2 arguments (batson_data, key), first argument not binary column"
            );
        };

        let Some(ColumnarValue::Scalar(ScalarValue::Utf8(Some(needle)))) = args.get(1) else {
            return exec_err!(
                "batson_contains expects 2 arguments (batson_data, key), got 1 argument"
            );
        };

        let mut result: Vec<bool> = vec![false; batson_column.len()];

        let path: Vec<batson::get::BatsonPath> = vec![needle.as_str().into()];

        for (index, opt_data) in batson_column.iter().enumerate() {
            if let Some(batson_data) = opt_data {
                match batson::get::contains(batson_data, &path) {
                    Ok(true) => {
                        result[index] = true;
                    }
                    Err(e) => {
                        return exec_err!("error decoding batson data: {:?}", e);
                    }
                    _ => {}
                }
            }
        }
        Ok(ColumnarValue::Array(Arc::new(BooleanArray::from(result))))
    }
}

async fn create_or_load<F, Fut>(
    ctx: &SessionContext,
    table_name: &str,
    generate: F,
) -> Result<(), ArrowError>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<RecordBatch, ArrowError>> + Send + 'static,
{
    let file_name = format!("{table_name}_{ROWS}_{COMPRESSION}.parquet");
    if Path::new(&file_name).exists() {
        println!("loading data from {}", &file_name);

        let options = ParquetReadOptions::default();
        ctx.register_parquet(&table_name, &file_name, options)
            .await?;
    } else {
        println!("creating data for {}", table_name);
        let start = Instant::now();
        let batch = generate().await?;

        ctx.register_batch(&table_name, batch)?;
        let df = ctx.table(table_name).await?;
        df.write_parquet(&file_name, DataFrameWriteOptions::default(), None)
            .await?;

        println!("created data for {} in {:?}", table_name, start.elapsed());
    }
    Ok(())
}

async fn create_table() -> Result<RecordBatch, ArrowError> {
    let mut rng = thread_rng();

    let mut id_builder = UInt64Builder::with_capacity(ROWS);
    let mut json_builder = LargeStringBuilder::with_capacity(ROWS, ROWS * 100);
    let mut batson_builder = BinaryBuilder::with_capacity(ROWS, ROWS * 100);

    for row in 0..ROWS {
        id_builder.append_value(row as u64);
        let json = random_json(row, &mut rng);
        json_builder.append_value(&json);
        let value = jiter::JsonValue::parse(json.as_bytes(), false).unwrap();
        let batson_data = batson::encode_from_json(&value).unwrap();
        batson_builder.append_value(&batson_data);
    }

    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("json", DataType::LargeUtf8, true),
            Field::new("batson", DataType::Binary, true),
        ])),
        vec![
            Arc::new(id_builder.finish()),
            Arc::new(json_builder.finish()),
            Arc::new(batson_builder.finish()),
        ],
    )
}

fn random_json(row: usize, rng: &mut ThreadRng) -> String {
    let mut json = String::new();
    json.push_str("{");
    // for _ in 0..rng.gen_range(2..=6) {
    for i in 0..rng.gen_range(5..=40) {
        if json.len() > 1 {
            json.push_str(",");
        }
        json.push('"');
        let key = COMMON_NAMES[rng.gen_range(0..COMMON_NAMES.len())];
        json.push_str(key);
        json.push_str(r#"": "#);
        json.push_str(&i.to_string());
    }
    let key = format!("row{}", row);
    json.push_str(&format!(r#","{key}": {row}"#));
    json.push('}');
    json
}

// fn random_string(rng: &mut ThreadRng) -> String {
//     (0..rng.gen_range(1..=50))
//         .map(|_| rng.sample(rand::distributions::Alphanumeric) as char)
//         .collect()
// }

const COMMON_NAMES: [&str; 200] = [
    "service.name",
    "service.namespace",
    "service.version",
    "service.instance.id",
    "telemetry.sdk.name",
    "telemetry.sdk.language",
    "telemetry.sdk.version",
    "http.method",
    "http.url",
    "http.target",
    "http.host",
    "http.scheme",
    "http.status_code",
    "http.status_text",
    "http.flavor",
    "http.server_name",
    "http.client_ip",
    "http.user_agent",
    "http.request_content_length",
    "http.request_content_length_uncompressed",
    "http.response_content_length",
    "http.response_content_length_uncompressed",
    "http.route",
    "http.client_header",
    "http.server_header",
    "db.system",
    "db.connection_string",
    "db.user",
    "db.name",
    "db.statement",
    "db.operation",
    "db.instance",
    "db.url",
    "db.sql.table",
    "db.cassandra.keyspace",
    "db.cassandra.page_size",
    "db.cassandra.consistency_level",
    "db.cassandra.table",
    "db.cassandra.idempotence",
    "db.cassandra.speculative_execution_count",
    "db.cassandra.coordinator_id",
    "db.cassandra.coordinator_dc",
    "db.hbase.namespace",
    "db.redis.database_index",
    "db.mongodb.collection",
    "db.sql.dml",
    "db.sql.primary_key",
    "db.sql.foreign_key",
    "db.sql.index_name",
    "rpc.system",
    "rpc.service",
    "rpc.method",
    "rpc.grpc.status_code",
    "net.transport",
    "net.peer.ip",
    "net.peer.port",
    "net.peer.name",
    "net.peer.hostname",
    "net.peer.address_family",
    "net.peer.ip_version",
    "net.host.ip",
    "net.host.port",
    "net.host.name",
    "net.protocol.name",
    "net.protocol.version",
    "net.destination.ip",
    "net.destination.port",
    "net.destination.name",
    "net.destination.subnet",
    "net.host.connection.type",
    "net.host.connection.subtype",
    "net.host.captured.ip",
    "net.host.captured.port",
    "net.host.creator",
    "net.destination.dns",
    "net.source.captured.ip",
    "net.source.captured.port",
    "net.source.creator",
    "messaging.system",
    "messaging.destination",
    "messaging.destination_kind",
    "messaging.protocol",
    "messaging.protocol_version",
    "messaging.url",
    "messaging.message_id",
    "messaging.conversation_id",
    "messaging.payload_size",
    "messaging.payload_compressed_size",
    "exception.type",
    "exception.message",
    "exception.stacktrace",
    "exception.escaped",
    "event.name",
    "event.domain",
    "event.id",
    "event.timestamp",
    "event.dropped_attributes_count",
    "log.severity",
    "log.message",
    "log.record_id",
    "log.timestamp",
    "log.file.path",
    "log.file.line",
    "log.function",
    "metric.name",
    "metric.description",
    "metric.unit",
    "metric.value_type",
    "metric.aggregation",
    "span.id",
    "span.name",
    "span.kind",
    "span.start_time",
    "span.end_time",
    "span.status_code",
    "span.status_description",
    "span.dropped_attributes_count",
    "span.dropped_events_count",
    "span.dropped_links_count",
    "span.remote",
    "span.parent_span_id",
    "span.parent_trace_id",
    "tracer.name",
    "tracer.version",
    "trace.id",
    "trace.state",
    "host.id",
    "host.type",
    "host.image.name",
    "host.image.id",
    "host.image.version",
    "host.architecture",
    "host.os.type",
    "host.os.description",
    "host.os.version",
    "host.os.name",
    "host.process.id",
    "host.process.name",
    "host.process.command",
    "host.user.id",
    "host.user.name",
    "container.id",
    "container.name",
    "container.image.name",
    "container.image.tag",
    "k8s.pod.name",
    "k8s.pod.uid",
    "k8s.namespace.name",
    "k8s.node.name",
    "k8s.node.uid",
    "k8s.cluster.name",
    "k8s.container.name",
    "k8s.container.restart_count",
    "k8s.deployment.name",
    "k8s.statefulset.name",
    "k8s.daemonset.name",
    "k8s.job.name",
    "k8s.job.uid",
    "k8s.cronjob.name",
    "cloud.provider",
    "cloud.account.id",
    "cloud.region",
    "cloud.availability_zone",
    "cloud.platform",
    "cloud.service.name",
    "cloud.service.namespace",
    "cloud.service.instance.id",
    "cloud.instance.id",
    "cloud.instance.name",
    "cloud.machine.type",
    "faas.trigger",
    "faas.execution",
    "faas.id",
    "faas.name",
    "faas.version",
    "faas.instance",
    "faas.max_memory",
    "faas.execution_time",
    "faas.runtime",
    "faas.cold_start",
    "faas.timeout",
    "resource.type",
    "resource.attributes",
    "enduser.id",
    "enduser.role",
    "enduser.scope",
    "telemetry.source",
    "telemetry.destination",
    "telemetry.data_type",
    "telemetry.data_source",
    "telemetry.data_destination",
    "telemetry.data_state",
    "telemetry.data_id",
    "telemetry.action",
    "telemetry.resource",
    "telemetry.agent",
    "telemetry.version",
    "telemetry.status",
    "telemetry.config",
    "service.environment",
];
