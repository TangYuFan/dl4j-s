package dl4j;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;


/**
 * @desc : mnist数字识别
 * @auth : TYF
 * @date : 2019-05-30 - 16:32
 */
public class t_2 {

    //加载单幅图片(28*28*1)
    public static INDArray loadPic(String path) throws Exception{
        //读取图片
        NativeImageLoader loader = new NativeImageLoader(28, 28, 1);
        //转为矩阵
        INDArray image = loader.asMatrix(new File(path));
        //归一化
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);
        return image;
    }

    //加载多幅图片(28*28*1)
    public static DataSetIterator loadPics(String path) throws Exception{
        Random ro = new Random(1234);
        //样本父目录
        File dir = new File(path);
        //分割
        FileSplit s_dir = new FileSplit(dir, NativeImageLoader.ALLOWED_FORMATS, ro);
        //样本加载器(按照目录名称生成标签)
        ImageRecordReader record = new ImageRecordReader(28, 28, 1, new ParentPathLabelGenerator());
        //初始化样本
        record.initialize(s_dir);
        //样本迭代器
        DataSetIterator iter = new RecordReaderDataSetIterator(record, 50, 1, record.numLabels());
        //归一化
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(iter);
        iter.setPreProcessor(scaler);
        return iter;
    }

    //构建网络
    public static MultiLayerNetwork createNet(){

        int seed = 1234;
        int channels = 1;
        int outputNum = 10;
        int iterations = 1;

        //设定动态改变学习速率的策略，key表示小批量迭代到几次
        Map<Integer, Double> lrSchedule = new HashMap<>();
        lrSchedule.put(0, 0.06);
        lrSchedule.put(200, 0.05);
        lrSchedule.put(600, 0.028);
        lrSchedule.put(800, 0.0060);
        lrSchedule.put(1000, 0.001);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                //初始权重
                .seed(seed)
                //迭代次数
                .iterations(iterations)
                //正则化
                .regularization(true).l2(0.0005)
                //学习速率
                .learningRate(.01)
                .learningRateDecayPolicy(LearningRatePolicy.Schedule)
                .learningRateSchedule(lrSchedule)
                //权重初始化方式
                .weightInit(WeightInit.XAVIER)
                //随机梯度下降
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //动量
                .updater(Updater.NESTEROVS)
                //添加多层网络
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //输出节点数
                        .nIn(channels)
                        .stride(1, 1)
                        //输出节点数
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .backprop(true).pretrain(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(10));

        //配置可视化
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        net.setListeners(new StatsListener(statsStorage));

        return net;
    }

    //训练
    private static void train(MultiLayerNetwork net,DataSetIterator trainIter,DataSetIterator testIter) throws Exception {
        int nEpochs = 1;
        for (int i = 0; i < nEpochs; i++) {
            System.out.println("fit start !");
            net.fit(trainIter);
            System.out.println("fit over !");
            //评价
            //Accuracy:        0.9906
            //Precision:       0.9906
            //Recall:          0.9905
            //F1 Score:        0.9906
            System.out.println(""+net.evaluate(testIter).stats());
            trainIter.reset();
            testIter.reset();
        }
        //保存网络
        ModelSerializer.writeModel(net, new File("./target/minist-model.zip"), true);
    }



    //测试
    private static void test(INDArray testData) throws Exception{
        MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(new File("./target/minist-model.zip"));
        System.out.println("result:"+net.predict(testData)[0]);
    }


    public static void main(String[] args) throws Exception{

        //训练
        //E:\work\dl4j\mnist_png\training
        DataSetIterator trainIterator = loadPics("D:\\my_dl4j\\mnist_png\\training");
        //E:\work\dl4j\mnist_png\testing
        DataSetIterator testIterator = loadPics("D:\\my_dl4j\\mnist_png\\testing");
        MultiLayerNetwork net = createNet();
        train(net,trainIterator,testIterator);
        //测试
        //E:\work\dl4j\testpic\8.bmp
        INDArray testData = loadPic("D:\\my_dl4j\\mnist_png\\test\\4.png");
        test(testData);
    }

}
