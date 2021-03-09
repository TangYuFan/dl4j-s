package dl4j;


import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

/**
 * @desc : 迁移学习，使用VGG构建图片分类模型
 * @auth : TYF
 * @data : 2019/6/16 17:32
 */
public class vggImageNet {


    //vgg是在Very Deep Convolutional Networks for Large-Scale Image Recognition期刊上提出的。
    //模型可以达到92.7%的测试准确度,在ImageNet的前5位。它的数据集包括1400万张图像，1000个类别。


    protected static long seed = 1234;
    protected static int height = 224;
    protected static int width = 224;
    protected static int channels = 3;
    protected static int numLabels = 2;
    protected static int batchSize = 20;
    protected static int nEpochs = 10;
    protected static List<String> labels ;
    protected static ComputationGraph net;


    //网络初始化
    public static void initNet() throws Exception{
        //用imageNet的权重初始化一个VGG16预训练模型
        ComputationGraph pretrained = (ComputationGraph) VGG16.builder().build().initPretrained(PretrainedType.IMAGENET);
        System.out.println("model:"+pretrained.summary());
        //修改部分配置
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .seed(1234)
                .build();
        //修改输出层
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                //冻结包含f2c之前所有层的权重
                .setFeatureExtractor("fc2")
                //删除原来的输出层
                .removeVertexKeepConnections("predictions")
                //添加新的输出层拼接到fc2后面
                .addLayer("predictions", new OutputLayer
                                .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096)
                                //标签个数
                                .nOut(numLabels)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX).build(), "fc2")
                .build();
        //可训练参数直接从138357544减少到40970个
        //Total Parameters:  134301514
        //Trainable Parameters:  40970
        //Frozen Parameters:  134260544
        System.out.println("model:"+vgg16Transfer.summary());
        vgg16Transfer.setListeners(new ScoreIterationListener(100));//100张计数一次
        net = vgg16Transfer;
    }

    //加载样本
    public static DataSetIterator loadPics(String path) throws Exception{
        Random ro = new Random(seed);
        FileSplit s_dir = new FileSplit(new File(path), NativeImageLoader.ALLOWED_FORMATS, ro);
        //样本加载器(按照目录名称生成标签)
        ImageRecordReader record = new ImageRecordReader(height, width ,channels, new ParentPathLabelGenerator());
        record.initialize(s_dir);
        DataSetIterator iter = new RecordReaderDataSetIterator(record, batchSize, 1, record.numLabels());
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(iter);
        iter.setPreProcessor(scaler);
        return iter;
    }
    //加载样本
    public static INDArray loadPic(String path) throws Exception{
        NativeImageLoader loader = new NativeImageLoader(height, width ,channels);
        INDArray image = loader.asMatrix(new File(path));
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);
        return image;
    }

    //训练和评估
    public static void train() throws Exception{
        initNet();
        //训练样本
        DataSetIterator train = loadPics(new File("").getCanonicalPath()+"/JPEGImages/train");//训练接猫狗各1w
        //测试样本
        DataSetIterator test = loadPics(new File("").getCanonicalPath()+"/JPEGImages/eval");//验证集猫狗各0.25w
        //评估
        Evaluation eval = net.evaluate(test);
        System.out.println("eval.stats():");
        System.out.println(eval.stats());
        System.out.println("train ...");
        test.reset();
        //训练
        for(int i=0;i<nEpochs;i++){
            net.fit(train);
            eval = net.evaluate(test);
            System.out.println("nEpochs:"+i+"   eval.stats():"+eval.stats());
            test.reset();
        }
        //标签
        labels = train.getLabels();
        //保存模型
        ModelSerializer.writeModel(net, new File("").getCanonicalPath()+"/model.zip", true);
        //关机
        Runtime.getRuntime().exec("shutdown -s -t 30");
    }

    //预测
    public static void predict() throws Exception{

        //窗口显示预测结果
        CanvasFrame canvas = new CanvasFrame("pic", 1);
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        //权重文件
        ComputationGraph model = ModelSerializer.restoreComputationGraph(new File("").getCanonicalPath()+"/model.zip");
        File[] files = new File(new File("").getCanonicalPath()+"/JPEGImages/test").listFiles();//测试集共0.1w
        for(int i=0;i<files.length;i++){
            //图片路径
            String in = files[i].getPath();
            //预测
            INDArray test = loadPic(in);
            INDArray result = model.outputSingle(false,test);
            System.out.println("result:"+result);
            //预测结果
            if( result.getDouble(0,0) < result.getDouble(0,1)){
                canvas.setTitle("狗 dog");
            }else{
                canvas.setTitle("猫 cat");
            }
            //原图显示
            opencv_core.Mat image = imread(in, IMREAD_COLOR);
            canvas.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            canvas.showImage(converter.convert(image));
            Thread.sleep(1500);
            if(i==(files.length-1)){
                i=0;
            }
        }
    }


    public static void main(String[] args) throws Exception{

        //train();
        predict();

        //Accuracy	准确率：模型准确识别出的MNIST图像数量占总数的百分比。
        //Precision	精确率：真正例的数量除以真正例与假正例数之和。
        //Recall	召回率：真正例的数量除以真正例与假负例数之和。
        //F1 Score	F1值：精确率和召回率的加权平均值。

        //精确率、召回率和F1值衡量的是模型的相关性。
        //举例来说，“癌症不会复发”这样的预测结果（即假负例/假阴性）就有风险，因为病人会不再寻求进一步治疗。
        //所以，比较明智的做法是选择一种可以避免假负例的模型（即精确率、召回率和F1值较高），尽管总体上的准确率可能会相对较低一些。

    }

}
