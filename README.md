## 군집(Clustering)?
   - 패턴 공간에 주어진 유한 개의 패턴들이 서로 가깝게 모여서 무리를 이루고 있는 패턴 집합을 묶는 과정.

## K-MEANS (KMEANS)란?
   - 주어진 데이터를 k개의 군집(클러스터:Clustering)로 묶는 알고리즘.
   - 각 클러스터와 거리 차이의 분산을 최소화하는 방식으로 동작.
   - 거리에 기반을 둔 clustering 기법
   - 기준점에 가까운 곳의 데이터들을 하나의 군집으로 묶는 방법.
   - 비지도학습 : Unsupervised Learning - [참고](https://ko.wikipedia.org/wiki/비_지도_학습)

## K-MEANS 수행과정
  - 임의의 "K개 중심값" 설정.
  - 전체 데이터와 "K개 중심값"을 비교, 가장 가까운 군집(K)에 소속.
  - 군집된 데이터를 기준으로 군집중앙의 위치를 제 설정.
  - 새롭개 구한 "k개 중심값"이 기존과 동일하면 알고리즘 종료.
  :: 이 과정을 통하여 K개의 군집으로 데이터를 구분.

##  Language별 구현체
   * Java
     - [Apache Commons Math](http://commons.apache.org/proper/commons-math/download_math.cgi) - [Overview](http://commons.apache.org/proper/commons-math/userguide/ml.html)

     - [WEKA 데이터마이닝 분석패키지 ( GNU )](https://www.cs.waikato.ac.nz/ml/weka/) : https://blog.naver.com/zxy826/220732514990

     - [Java-ML](http://java-ml.sourceforge.net/content/installing-library) : Java Machine Learning Library
     - [xetorthio/kmeans](https://github.com/xetorthio/kmeans/tree/master/src)

   * R
     kmeans() 함수 이용.
     - 첫 번째 인자: 군집에 사용할 데이터
     - 두 번째 인자: 클러스터의 수(k)

   * Javascript
     - nodejs : [nodeml](https://www.npmjs.com/package/nodeml) : node machine learning
     - 그외 참고 : [javascript kmeans](https://github.com/mlehman/kmeans-javascript)

## 기타 참고:
   - https://github.com/xetorthio/kmeans/tree/master/src
   - http://action713.tistory.com/entry/K평균-알고리즘Kmeans-algorithm
   - http://datamining.uos.ac.kr/wp-content/uploads/2016/09/10장-클러스터링.pdf
   - https://blog.naver.com/pjc1349/20057343166
   - EM Clustering : https://blog.naver.com/pjc1349/20064718100
   - https://www.nextobe.com/single-post/2018/02/26/데이터-과학자가-알아야-할-5가지-클러스터링-알고리즘

## K-Means (5) comparison with EM
   - 원본 : http://ai-times.tistory.com/158
   - K-Means
     - Hard Clustering : A instance belong to only one Cluster.
     - Based on Euclidean distance.
     - Not Robust on outlier, value range.

   - EM
     - Soft Clustering : A instance belong to several clusters with
     - membership probability.
     - Based on density probability.  
     - Can handle both numeric and nominal attributes.

## source - nodeml (nodejs)
```javascript
'use strict';

const {sample, kMeans, evaluate} = require('nodeml');
const bulk = sample.iris();

let kmeans = new kMeans();
kmeans.train(bulk.dataset, {k: 3});
result = knn.test(bulk.dataset);

console.log(result);
```
## source - javascript 구현소스 (원본 : https://proinlab.com/archives/2134)
  1. [kmeans-sample-test-01-Find Centroid](https://scrimba.com/c/cPwp3hZ)
  2. [kmeans-sample-test-02-Euclidean Distance](https://scrimba.com/c/cb3ZJHa)
  3. [kmeans-sample-test-03-Expectation](https://scrimba.com/c/cvLVvsn)
  4. [kmeans-sample-test-04-Maximazation](https://scrimba.com/c/czZV4Hd)
  5. [kmeans-sample-test-05-왜곡 측정 및 Iteration](https://scrimba.com/c/cEaDPuK)  

## source - java : https://github.com/xetorthio/kmeans
// todo

## source - java : http://ai-times.tistory.com/158
 ```java
 package xminer.mining.clustering;

 import xminer.core.*;
 import java.util.Random;

 public class KMeans {
   Dataset m_dataset;
   int m_clusterSize;
   int m_maxIteration;
   int m_recordCount;
   int m_fieldCount;
   int m_recordClusterIndex[];   // 각 레코드에 대하여 소속 군집번호
   int m_clusterCount[];            // 각 클러스터별 소속 개수
   Record m_cetroids[];

   public KMeans(Dataset ds, int clusterSize, int maxIteration) {
     m_dataset = ds;
     this.m_clusterSize = clusterSize;
     this.m_maxIteration = maxIteration;
     this.m_recordCount = ds.getRecordCount();
     this.m_fieldCount = ds.getAttrCount();
     this.m_recordClusterIndex = new int[ ds.getRecordCount() ];
     this.m_cetroids = new Record[ this.m_clusterSize ];
     this.m_clusterCount = new int[ this.m_clusterSize ];
   }

   public void learn(){
     // 초기 랜덤 시드 결정
     int i=0;
     init_centroid();
     this.print_centroid();

     while(true){
       //System.out.println( i + "번재 수행결과");

       reAssign_Step();
       findNewCentroid_Step();

       // System.out.println();
       // this.print_centroid();
       // this.print_state();

       // 최대반복횟수에 의한 학습 종료
       i++;
       if( i >= this.m_maxIteration){
         System.out.println("최대반복횟수에 도달하여 종료합니다. 반복횟수 : " + i);
         break;
       }

       // 중심점(Centroid)의 고정에 의한 학습 종료
       // -- 새로운 중심점의 계산
       // -- 이전 중심점과의 차이를 계산
       // -- 만약 중심점의 변화가 없으면 끝

     }
     System.out.println( i + "번재 수행결과");
     System.out.println();
     this.print_centroid();
     this.print_state();

   }

   /**
    * 초기에 클러스터 개수만큼의 레코드를 선택하여 이들을 초기 군집 중심으로 합니다.
    * 이때 같은 레코드가 중복해서 다른 군집의 중심점이 되지 않도록 합니다.
    */
   public void init_centroid(){
     Random random = new Random();
     for(int c=0; c<this.m_clusterSize; c++){
       this.m_cetroids[c] = this.m_dataset.getRecord( random.nextInt(m_recordCount-1)).copy();
     }
   }

   /**
    * 군집의 중심을 새롭게 계산합니다.
    * 모든 레코드의 소속값을 고려하여 평균값을 정합니다.
    */
   public void findNewCentroid_Step(){
     // 초기화
     for(int c=0; c<this.m_clusterSize; c++){
       this.m_clusterCount[c] = 0;
       for(int f=0; f<this.m_fieldCount; f++){
        this.m_cetroids[c].add(f, 0.0);
       }
     }
     int c_num;
     // 클러스터별 소속 레코드 개수를 계산합니다.
     for(int r=0; r<this.m_recordCount; r++){
       c_num = this.m_recordClusterIndex[r];
       this.m_clusterCount[c_num]++;
     }
     // 클러스터별 중심을 계산합니다.
     for(int r=0; r<this.m_recordCount; r++){
       c_num = this.m_recordClusterIndex[r];
       Record record = this.m_dataset.getRecord(r).copy();
       for(int f=0; f<this.m_fieldCount; f++){
        this.m_cetroids[c_num].addOnPrevValue(f, record.getValue(f));
       }
     }
     for(int c=0; c<this.m_clusterSize; c++){
       //System.out.println("군집 " + c + "의 개수 : "  + this.m_clusterCount[c]);
       this.m_cetroids[c].multiply( 1.0/(double)this.m_clusterCount[c] );
     }
   }

   /**
    * 주어진 중심에 대하여 모든 레코드들을 지정(Assign)합니다.
    * 레코드와 각 군집중심과의 거리를 계산해보고 가장 거리가 가까운 군집에 지정합니다.
    */
   public void reAssign_Step(){
     int c_num;
     double min_dist = Double.POSITIVE_INFINITY;
     double distance;
     for(int r=0; r<this.m_recordCount; r++){
       Record record = this.m_dataset.getRecord(r).copy();
       c_num = 0;
       min_dist = 10000; //Double.POSITIVE_INFINITY;
       for(int c=0; c<this.m_clusterSize; c++){
         distance = this.m_dataset.getDistanceOfUclideanP(record, this.m_cetroids[c]);
         // 해당 레코드와 군집중심과의 거리를 계산합니다.
         if(distance < min_dist){ // 최소
           min_dist = distance;
           c_num = c;
         }
       }
       this.m_recordClusterIndex[r] = c_num;
     }
   }

   /**
    * 현재 중심점(Centroid)의 값을 출력합니다.
    */
   public void print_centroid() {
     for (int c = 0; c < this.m_clusterSize; c++) {
       System.out.println("군집[" + (c) + "]의 중심점 : " +  this.m_cetroids[c].toString());
     }
   }

   public void print_state(){
     for(int r=0; r<this.m_recordCount; r++){
       System.out.print("번호 "+ (r+1) + " : " );
       for(int f=0; f<this.m_fieldCount; f++){
         System.out.print( this.m_dataset.getRecord(r).getValue(f) + ", " );
       }
       System.out.println( this.m_recordClusterIndex[r] );
     }
   }

   public static void main(String[] args) {
     // 아이리스 원본 데이터
     Dataset ds = new Dataset("아이리스","D:\\ai-miner-test-data\\iris.csv",  

                        Dataset.FIRSTLINE_ATTR_NO_INFO, true);
     // 수치 필드만 있는 아이리스 데이터
     // Dataset ds = new Dataset("D:\\work\\(01) (입력모듈) Dataset\\datafile\\iris_4.csv",

                            Dataset.HAS_NOT_FIELD_NAME_LINE);

     ds.printDataInfo();
     KMeans km = new KMeans(ds, 3, 200); // 3개 군집, 최대 10번 반복, 종료변화값 0.01
     km.learn();
   }

 }
// 출처: http://ai-times.tistory.com/158 [ai-times]
 ```
