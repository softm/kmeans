## 군집(Clustering)?
   - 패턴 공간에 주어진 유한 개의 패턴들이 서로 가깝게 모여서 무리를 이루고 있는 패턴 집합을 묶는 과정.

## K-MEANS (KMEANS)란?
   - [K-평균알고리즘 - WIKI](https://ko.wikipedia.org/wiki/K-평균_알고리즘)

   - 주어진 데이터를 k개의 군집(클러스터:Clustering)로 묶는 알고리즘.
   - 각 클러스터와 거리 차이의 분산을 최소화하는 방식으로 동작.
   - 거리에 기반을 둔 clustering 기법
   - 기준점에 가까운 곳의 데이터들을 하나의 군집으로 묶는 방법.
   - 비지도학습 : Unsupervised Learning - [참고](https://ko.wikipedia.org/wiki/비_지도_학습)

## K-MEANS 수행과정1
  1. 임의의 "K개 중심값" 설정.
  2. 전체 데이터와 "K개 중심값"을 비교, 가장 가까운 군집(K)에 소속.
  3. 군집된 데이터를 기준으로 군집중앙의 위치를 제 설정.
  4. 새롭개 구한 "k개 중심값"이 기존과 동일하면 알고리즘 종료.
     :: 이 과정을 통하여 K개의 군집으로 데이터를 구분.

# 수행과정 - javascript
  1. 초기 중심값(init Centroid) 초기화.
     * 알고리즘 : https://ko.wikipedia.org/wiki/K-평균_알고리즘#초기화_기법
        - Random Partition
        - Forgy, MacQueen
        - Kaufman
  2. Data간 거리 계산
     1. 초기 중심값(init Centroid) 과 Data간 거리 계산.
        * [유클리드 거리(Euclidean distance)](https://ko.wikipedia.org/wiki/유클리드_거리)
        ```JavaScript
        /* 초기 중심값과 Data의 거리 비교.
          - 초기 중심값 : center
          - Data       : dataset
          >> distance(dataset[n], center[k]);
        */
        let c = [];
        for(let n = 0 ; n < dataset.length ; n++) {
            let x = dataset[n];
            let minDist = -1, rn = 0;
            for(let k = 0 ; k < center.length ; k++) {
                let dist = distance(dataset[n], center[k]);
                if(minDist === -1 || minDist > dist) {
                    minDist = dist;
                    cn = k;
                }
            }
            c[n] = cn;
        }        
        // c : 클러스터링 분류정보 (0~K)
        ```

     2. 거리가 가까운 군집에 Data를 분류.
        ```JavaScript
        c[n] = rn;
        ```        
     3. 중심점 최적화 및 왜곡측정.
        * 왜곡측정 : 거리의 합을 비교.
        1. 중심점 계산 :centroid();
        2. 왜곡측정 : 중심점과 Data간 최소거리의합과 이전최소거리의 합의 변화를 비교함.
        ```JavaScript
        let preJ = 0;
        while(true) {
            let c = centroid();

            // 왜곡측정 : 거리의 합을 이용.
            let J = 0;
            for(let n = 0 ; n < dataset.length ; n++) {
                let x = dataset[n];
                let minDist = -1;
                for(let k = 0 ; k < center.length ; k++) {
                    let dist = distance(dataset[n], center[k]);
                    if(minDist === -1 || minDist > dist) {
                        minDist = dist;
                    }
                }
                J += minDist;
            }

            // 이전값과 비교하여 차이가 없으면 종료
            let diff = Math.abs(preJ - J);
            if(diff <= 0) {
                console.info("last-cluster",c);        
                break;
            } else {
                //debugger;
                //console.info(diff,preJ,J);
            }
            preJ = J;
        };

        function centroid() {
            let c = [];
            for(let n = 0 ; n < dataset.length ; n++) {
                let x = dataset[n];
                let minDist = -1, cn = 0;
                for(let k = 0 ; k < center.length ; k++) {
                    let dist = distance(dataset[n], center[k]);
                    if(minDist === -1 || minDist > dist) {
                        minDist = dist;
                        cn = k;
                    }
                }
                c[n] = cn;
            }

            //center null 초기화
            center = Array.apply(null, Array(center.length));
            let clusterCount = Array.apply(null, Array(center.length)).map(Number.prototype.valueOf, 0);

            for(let n = 0 ; n < dataset.length ; n++) {
                let k = c[n] * 1;
                let x = dataset[n];

                if(!center[k]) center[k] = {};

                //for(let key in x) {
                for ( let key = 0;key<x.length; key++) {
                    if(!center[k][key]) center[k][key] = 0;
                    center[k][key] += x[key] * 1;
                }
                //console.info("clusterCount["+k+"]",clusterCount[k]);
                clusterCount[k]++;
            }

            for(let k = 0 ; k < center.length ; k++) {
                for(let _key in center[k]) {
                    center[k][_key] = center[k][_key] / clusterCount[k * 1];
                    //console.info("center["+k+"]["+_key+"]",center[k][_key]);        
                }
            }
            //console.info("re---center",center);
            return c;
        }
        ```
   4. 왜곡이 있는동안 "3.중심점 최적화 및 왜곡측정." 반복 수행.

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

## source - javascript 구현소스 (원본 : https://proinlab.com/archives/2134)
   1. [kmeans-sample-test-01-Find Centroid](https://scrimba.com/c/cPwp3hZ)
   2. [kmeans-sample-test-02-Euclidean Distance](https://scrimba.com/c/cb3ZJHa)
   3. [kmeans-sample-test-03-Expectation](https://scrimba.com/c/cvLVvsn)
   4. [kmeans-sample-test-04-Maximazation](https://scrimba.com/c/czZV4Hd)
   5. [kmeans-sample-test-05-왜곡 측정 및 Iteration](https://scrimba.com/c/cEaDPuK)  

```javascript
function kmeans(k,dataset) {
    let center = [];
    let preRand = {}; // 중복된 center가 존재하지 않도록 점검
    while(true) { // k개의 데이터가 선택될 때까지 실행
        let rand = Math.floor(Math.random() * dataset.length);
        if(preRand[rand]) continue;
        if(dataset[rand]) {
            center.push(dataset[rand]);
            preRand[rand] = true;
        }
        if(center.length == k) break;
    }
    //center.sort();
    // console.log("init-center",center);
    center[5,15,25];
    let distance = (x, y)=> {
        let sum = 0;
        let keys = {};
        for(let key in x) keys[key] = true;
        for(let key in y) keys[key] = true;
        //console.info("keys",keys);
        for(let key in keys) {
            let xd = x[key] ? x[key] * 1 : 0;
            let yd = y[key] ? y[key] * 1 : 0;
            sum += (xd - yd) * (xd - yd);
        }

        return Math.sqrt(sum);
    };

    //console.info(distance([1,2,3],[1]));
    //console.info("init-cluster-let r",r)
    function centroid() {
        let c = [];
        for(let n = 0 ; n < dataset.length ; n++) {
            let x = dataset[n];
            let minDist = -1, cn = 0;
            for(let k = 0 ; k < center.length ; k++) {
                let dist = distance(dataset[n], center[k]);
                if(minDist === -1 || minDist > dist) {
                    minDist = dist;
                    cn = k;
                }
            }
            c[n] = cn;
        }

        //center null 초기화
        center = Array.apply(null, Array(center.length));
        let clusterCount = Array.apply(null, Array(center.length)).map(Number.prototype.valueOf, 0);

        for(let n = 0 ; n < dataset.length ; n++) {
            let k = c[n] * 1;
            let x = dataset[n];

            if(!center[k]) center[k] = {};

            //for(let key in x) {
            for ( let key = 0;key<x.length; key++) {
                if(!center[k][key]) center[k][key] = 0;
                center[k][key] += x[key] * 1;
            }
            //console.info("clusterCount["+k+"]",clusterCount[k]);
            clusterCount[k]++;
        }

        for(let k = 0 ; k < center.length ; k++) {
            for(let _key in center[k]) {
                center[k][_key] = center[k][_key] / clusterCount[k * 1];
                //console.info("center["+k+"]["+_key+"]",center[k][_key]);        
            }
        }
        //console.info("re---center",center);
        return c;
    }
    //////////////// //////////////// //////////////// //////////////// ////////////////

    centroid();

    let preJ = 0;
    let c = [];
    while(true) {
        c = centroid();

        // 왜곡측정 : 거리의 합을 이용.
        let J = 0;
        for(let n = 0 ; n < dataset.length ; n++) {
            let x = dataset[n];
            let minDist = -1;
            for(let k = 0 ; k < center.length ; k++) {
                let dist = distance(dataset[n], center[k]);
                if(minDist === -1 || minDist > dist) {
                    minDist = dist;
                }
            }
            J += minDist;
        }

        // 이전값과 비교하여 차이가 없으면 종료
        let diff = Math.abs(preJ - J);
        if(diff <= 0) {
            console.info("last-cluster",c);        
            break;
        } else {
            //debugger;
            //console.info(diff,preJ,J);
        }
        preJ = J;
    };
    return c;
}

let dataset = [
    [	1	,				]	,
	[	2	,				]	,
	[	3	,				]	,
	[	4	,				]	,
	[	5	,				]	,
	[	6	,				]	,
	[	14	,				]	,
	[	15	,				]	,
	[	16	,				]	,
	[	17	,				]	,
	[	18	,				]	,
	[	19	,				]	,
	[	20	,				]	,
	[	21	,				]	,
	[	22	,				]	,
	[	23	,				]	,
	[	24	,				]	,
	[	25	,				]	,
	[	26	,				]	,
	[	27	,				]	,
	[	28	,				]	,
	[	7	,				]	,
	[	8	,				]	,
	[	9	,				]	,
	[	10	,				]	,
	[	11	,				]	,
	[	12	,				]	,
	[	13	,				]	,    
	[	29	,				]	,
	[	30					]
];

for (var i=0;i<1;i++) {
    let c = kmeans(3,dataset);
    let v0 = c.map(function(r,idx){
        //console.info(r,idx);
        if ( r == 0 ) {
            return dataset[idx][0];
        }
    }).filter(function(data) {return data !=null} );
    console.info("cluster 0", v0);
    let v1 = c.map(function(r,idx){
        //console.info(r,idx);
        if ( r == 1 ) {
            return dataset[idx][0];
        }
    }).filter(function(data) {return data !=null} );
    console.info("cluster 1", v1);
    let v2 = c.map(function(r,idx){
        //console.info(r,idx);
        if ( r == 2 ) {
            return dataset[idx][0];
        }
    }).filter(function(data) {return data !=null} );  
    console.info("cluster 2", v2);
}
//kmeans(3,dataset);
```

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
## source - java
- [Apache Commons Math](http://commons.apache.org/proper/commons-math/download_math.cgi) - [Overview](http://commons.apache.org/proper/commons-math/userguide/ml.html)

```java
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.Clusterer;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class TestClusterer
{
	@Test
	public void test1() throws Exception {
		for (int i=0;i<10;i++) {		
		Clusterer<DoublePoint> clusterer = new KMeansPlusPlusClusterer<DoublePoint>(3);
		List<DoublePoint> list = new ArrayList<DoublePoint>();

		list.add(new DoublePoint(new double[]{	1	}));
		list.add(new DoublePoint(new double[]{	2	}));
		list.add(new DoublePoint(new double[]{	3	}));
		list.add(new DoublePoint(new double[]{	4	}));
		list.add(new DoublePoint(new double[]{	5	}));
		list.add(new DoublePoint(new double[]{	6	}));
		list.add(new DoublePoint(new double[]{	7	}));
		list.add(new DoublePoint(new double[]{	8	}));
		list.add(new DoublePoint(new double[]{	9	}));
		list.add(new DoublePoint(new double[]{	10	}));
		list.add(new DoublePoint(new double[]{	11	}));
		list.add(new DoublePoint(new double[]{	12	}));
		list.add(new DoublePoint(new double[]{	13	}));
		list.add(new DoublePoint(new double[]{	14	}));
		list.add(new DoublePoint(new double[]{	15	}));
		list.add(new DoublePoint(new double[]{	16	}));
		list.add(new DoublePoint(new double[]{	17	}));
		list.add(new DoublePoint(new double[]{	18	}));
		list.add(new DoublePoint(new double[]{	19	}));
		list.add(new DoublePoint(new double[]{	20	}));
		list.add(new DoublePoint(new double[]{	21	}));
		list.add(new DoublePoint(new double[]{	22	}));
		list.add(new DoublePoint(new double[]{	23	}));
		list.add(new DoublePoint(new double[]{	24	}));
		list.add(new DoublePoint(new double[]{	25	}));
		list.add(new DoublePoint(new double[]{	26	}));
		list.add(new DoublePoint(new double[]{	27	}));
		list.add(new DoublePoint(new double[]{	28	}));
		list.add(new DoublePoint(new double[]{	29	}));
		list.add(new DoublePoint(new double[]{	30	}));

//		System.out.println(list);

		List<? extends Cluster<DoublePoint>> res = clusterer.cluster(list);
//		System.out.println("!!!");
//		System.out.println(res.size());
		//System.out.println(res.toString());
			int seq = 0;
			for (Cluster<DoublePoint> re : res) {
				System.out.print(re.getPoints() + " / ");
				seq++;
				if ( seq % 3 == 0) System.out.print("\n");				
			}
		}
	}
```
## source - java : https://github.com/xetorthio/kmeans

## source - java : http://ai-times.tistory.com/158
 ```java
 // 출처: http://ai-times.tistory.com/158 [ai-times]
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
```
