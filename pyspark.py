from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("ExampleApp") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "10g") \
    .getOrCreate()
data = [("Alice", 25), ("Bob", 30), ("Charlie", 28)]
columns = ["name", "age"]
df = spark.createDataFrame(data, columns)
df.show()