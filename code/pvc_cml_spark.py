DATALAKE_DIRECTORY = "hdfs://ns1"

spark = (SparkSession.builder.appName("MyApp")\
  .config("spark.jars", "/opt/spark/optional-lib/iceberg-spark-runtime.jar")\
  .config("spark.sql.hive.hwc.execution.mode", "spark")\
  .config( "spark.sql.extensions", "com.qubole.spark.hiveacid.HiveAcidAutoConvertExtension, org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")\
  .config("spark.sql.catalog.spark_catalog.type", "hive")\
  .config( "spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog")\
  .config("spark.yarn.access.hadoopFileSystems", DATALAKE_DIRECTORY)\
  .getOrCreate()
  )
