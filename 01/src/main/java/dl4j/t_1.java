package dl4j;


import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * @desc : MNIST字符识别样例 0.8.0版本
 * @auth : TYF
 * @date : 2019-05-30 - 14:49
 */
public class t_1 {

    private static Logger log = LoggerFactory.getLogger(t_1.class);

    public static void main(String[] args) throws Exception{
        //样本行
        final int numRows = 28;
        //样本列
        final int numColumns = 28;
        //标签行
        int outputNum = 10;
        //每次迭代抓取的样本个数
        int batchSize = 128;
        //初始权向量
        int rngSeed = 123;
        //迭代周期
        int numEpochs = 15;

        //训练数据
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        //测试数据
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)  //初始权向量
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)    //随机梯度下降
                .iterations(1)  //迭代(权向量更新)次数
                .learningRate(0.006)    //每次迭代权向量调整幅度
                .updater(Updater.NESTEROVS)     //权重更新器
                .regularization(true).l2(1e-4)      //正则化
                .list()     //网络层数
                .layer(0, new DenseLayer.Builder()      //DenseLayer代表一个全连接(连接到一个隐藏层)
                        .nIn(numRows * numColumns)           //输入样本
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)   //OutputLayer代表一个全连接(连接到一个输出层)
                        .nIn(1000)
                        .nOut(outputNum)                    //输出样本
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));

        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            model.fit(mnistTrain);
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");
    }


}
