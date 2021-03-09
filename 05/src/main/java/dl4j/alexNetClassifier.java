package dl4j;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.AlexNet;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import java.io.File;
import java.util.Arrays;
import java.util.Random;

/**
 * @desc : alexNet分类器 训练mnist数据
 * @auth : TYF
 * @date : 2019-06-25 - 15:35
 */
public class alexNetClassifier {

    //样本配置
    protected static int height = 224;
    protected static int width = 224;
    protected static int channels = 1;
    //网络配置
    protected static long seed = 1234;
    protected static IUpdater updater = new Nesterovs(1e-2, 0.9);
    protected static int[] inputShape = new int[] {channels, height, width};
    protected static int numLabels = 10;//标签数
    protected static int batchSize = 20;
    protected static WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    protected static CacheMode cacheMode = CacheMode.NONE;
    //训练配置
    protected static int nEpochs = 5;
    //网络
    protected static MultiLayerNetwork net;

    //网络初始化
    public static void init(){
        //模型
        AlexNet.AlexNetBuilder builder = AlexNet.builder();
        //设置参数
        builder.updater(updater);
        builder.inputShape(inputShape);
        builder.numClasses(numLabels);
        builder.cacheMode(cacheMode);
        builder.workspaceMode(workspaceMode);
        //网络
        ZooModel model = builder.build();
        net = model.init();
        net.setListeners(new ScoreIterationListener(1));
        System.out.println("net:"+net.summary());


    }


    //加载多幅图片
    public static DataSetIterator loadPics(String path) throws Exception{
        Random ro = new Random(seed);
        File dir = new File(new File("").getCanonicalPath()+"/JPEGImages");
        FileSplit s_dir = new FileSplit(dir, NativeImageLoader.ALLOWED_FORMATS, ro);
        //标签自动生成
        ImageRecordReader record = new ImageRecordReader(height, width ,channels, new ParentPathLabelGenerator());
        record.initialize(s_dir);
        DataSetIterator iter = new RecordReaderDataSetIterator(record, batchSize, 1, record.numLabels());
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(iter);
        iter.setPreProcessor(scaler);
        return iter;
    }

    //加载单幅图片
    public static INDArray loadPic(String path) throws Exception{
        NativeImageLoader loader = new NativeImageLoader(height, width ,channels);
        INDArray image = loader.asMatrix(new File(path));
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);
        return image;
    }

    //训练
    public static void train() throws Exception{
        //加载数据
        DataSetIterator train = loadPics(new File("").getCanonicalPath()+"/JPEGImages/training");//训练集
        DataSetIterator test = loadPics(new File("").getCanonicalPath()+"/JPEGImages/testing");//验证集
        //评估
        Evaluation eval ;
        test.reset();
        //训练
        for (int i = 0; i < nEpochs; i++) {
            train.reset();
            while (train.hasNext()) {
                net.fit(train.next());
            }
            System.out.println("Completed epoch:"+i);
            eval = net.evaluate(test);
            System.out.println("eval.stats():"+eval.stats());
            //保存日志
            Log_Exception.writeEror_to_txt(new File("").getCanonicalPath()+"/log.txt","Completed epoch:"+i);
            Log_Exception.writeEror_to_txt(new File("").getCanonicalPath()+"/log.txt","eval.stats():"+eval.stats());
            test.reset();
        }
        //保存网络
        ModelSerializer.writeModel(net, new File("").getCanonicalPath()+"/model.zip", true);
        //关机
        Runtime.getRuntime().exec("shutdown -s -t 30");
    }

    //预测
    public static void predict() throws Exception{
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("").getCanonicalPath()+"/model.zip");
        //预测
        INDArray test = loadPic("D:\\my_idea_workspace\\dl4j\\dl4j\\05\\JPEGImages\\testing\\5\\8.png");
        int[] result = model.predict(test);
        System.out.println("result:"+ Arrays.toString(result));
    }


    public static void main(String[] args) throws Exception{
        init();
        train();
        predict();
    }


}
