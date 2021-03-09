package dl4j;


import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;


/**
 * @desc : 用dl4j实现AlexNet网络结构
 * @auth : TYF
 * @data : 2019/6/16 14:13
 */
public class _alexNet_ {

    //样本配置
    protected static int height = 224;
    protected static int width = 224;
    protected static int channels = 3;
    //网络配置
    protected static MultiLayerNetwork network;
    protected static long seed = 1234;
    protected static IUpdater updater = new Nesterovs(1e-2, 0.9);
    protected static int[] inputShape = new int[] {channels, height, width};//输入参数(通道数、宽高)
    protected static int numLabels = 10;//输出标签个数
    protected static WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    protected static CacheMode cacheMode = CacheMode.NONE;
    //训练配置

    //网络初始化
    public static void netInit(){
        //以下是按照alexNet结构拼的网络，也是zoo中AlexNet源码的做法
        double nonZeroBias = 1;//偏差
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
                .weightInit(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(updater)
                .biasUpdater(new Nesterovs(2e-2, 0.9))
                .convolutionMode(ConvolutionMode.Same)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .cacheMode(cacheMode)
                .l2(5 * 1e-4)
                .miniBatch(false)
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{11,11}, new int[]{4, 4})
                        .name("cnn1")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .nIn(inputShape[0])
                        .nOut(96)
                        .build())
                .layer(1, new LocalResponseNormalization.Builder().build())
                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .padding(1,1)
                        .name("maxpool1")
                        .build())
                .layer(3, new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{1,1}, new int[]{2,2})
                        .name("cnn2")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3}, new int[]{2, 2})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .name("maxpool2")
                        .build())
                .layer(5, new LocalResponseNormalization.Builder().build())
                .layer(6, new ConvolutionLayer.Builder()
                        .kernelSize(3,3)
                        .stride(1,1)
                        .convolutionMode(ConvolutionMode.Same)
                        .name("cnn3")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .nOut(384)
                        .build())
                .layer(7, new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1})
                        .name("cnn4")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .nOut(384)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(8, new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1})
                        .name("cnn5")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .nOut(256)
                        .biasInit(nonZeroBias)
                        .build())
                .layer(9, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{2,2})
                        .name("maxpool3")
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build())
                .layer(10, new DenseLayer.Builder()
                        .name("ffn1")
                        .nIn(256*6*6)
                        .nOut(4096)
                        .weightInit(new NormalDistribution(0, 0.005))
                        .biasInit(nonZeroBias)
                        .build())
                .layer(11, new DenseLayer.Builder()
                        .name("ffn2")
                        .nOut(4096)
                        .weightInit(new NormalDistribution(0, 0.005))
                        .biasInit(nonZeroBias)
                        .dropOut(0.5)
                        .build())
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .weightInit(new NormalDistribution(0, 0.005))
                        .biasInit(0.1)
                        .build())


                .setInputType(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]))
                .build();
        network = new MultiLayerNetwork(conf);
        network.init();
    }


    public static void main(String[] args) {
        netInit();
        System.out.println("net:"+network.summary());
    }
}
