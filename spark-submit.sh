source /etc/profile
#your spark-env path
#source /usr/local/spark-2.0.0-bin-hadoop2.7/conf/spark-env.sh
# spark
spark-submit --class word2vec --master yarn-cluster --num-executors 20 --driver-memory 45g --executor-memory 10g --executor-cores 1 \ 
--conf spark.akka.frameSize=2046 --conf spark.driver.maxResultSize=10g \
--jars $cur_path/jars/scopt_2.10-3.2.0.jar \
 $cur_path/w2v.jar  \
 --input $1 \
 --output $2 \
 --size 600 \
 --window 8 \ 
 --alpha 0.025 \
 --minicount 3 \ 
 --iterations 1 \ 
 --partitions 20  
