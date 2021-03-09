package dl4j.yolo2;


import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SpaceToDepthLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Random;
import static org.deeplearning4j.zoo.model.helper.DarknetHelper.addLayers;

/**
*   @desc : yolo2源码网络拼接、通用目标检测。
*   @auth : TYF
*   @date : 2019-08-01 - 17:08
*/
public class yolo_v2 {

    //net
    private static final double[][] DEFAULT_PRIOR_BOXES = {{0.57273, 0.677385}, {1.87446, 2.06253}, {3.33843, 5.47434}, {7.88282, 3.52778}, {9.77052, 9.16828}};
    private static double[][] priorBoxes = DEFAULT_PRIOR_BOXES;
    private static long seed = 1234;
    private static IUpdater updater = new Adam(1e-3);
    private static CacheMode cacheMode = CacheMode.NONE;
    private static WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    private static ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;
    private static Random rng = new Random(seed);
    private static int nBoxes = 5;
    //train
    private static int numClasses = 2;
    private static int[] inputShape = {3,960,540};
    private static int gridWidth = 30;
    private static int gridHeight = 17;
    private static int nEpochs = 20;
    private static int batchSize = 2;

    /**
    *   @desc : 网络配置
    *   @auth : TYF
    *   @date : 2019-08-01 - 16:55
    */
    public static ComputationGraphConfiguration conf(){
        INDArray priors = Nd4j.create(priorBoxes);
        GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(updater)
                .l2(0.00001)
                .activation(Activation.IDENTITY)
                .cacheMode(cacheMode)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .cudnnAlgoMode(cudnnAlgoMode)
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]));
        addLayers(graphBuilder, 1, 3, inputShape[0],  32, 2);
        addLayers(graphBuilder, 2, 3, 32, 64, 2);
        addLayers(graphBuilder, 3, 3, 64, 128, 0);
        addLayers(graphBuilder, 4, 1, 128, 64, 0);
        addLayers(graphBuilder, 5, 3, 64, 128, 2);
        addLayers(graphBuilder, 6, 3, 128, 256, 0);
        addLayers(graphBuilder, 7, 1, 256, 128, 0);
        addLayers(graphBuilder, 8, 3, 128, 256, 2);
        addLayers(graphBuilder, 9, 3, 256, 512, 0);
        addLayers(graphBuilder, 10, 1, 512, 256, 0);
        addLayers(graphBuilder, 11, 3, 256, 512, 0);
        addLayers(graphBuilder, 12, 1, 512, 256, 0);
        addLayers(graphBuilder, 13, 3, 256, 512, 2);
        addLayers(graphBuilder, 14, 3, 512, 1024, 0);
        addLayers(graphBuilder, 15, 1, 1024, 512, 0);
        addLayers(graphBuilder, 16, 3, 512, 1024, 0);
        addLayers(graphBuilder, 17, 1, 1024, 512, 0);
        addLayers(graphBuilder, 18, 3, 512, 1024, 0);
        // #######
        addLayers(graphBuilder, 19, 3, 1024, 1024, 0);
        addLayers(graphBuilder, 20, 3, 1024, 1024, 0);
        // route
        addLayers(graphBuilder, 21, "activation_13", 1, 512, 64, 0, 0);
        // reorg
        graphBuilder.addLayer("rearrange_21",new SpaceToDepthLayer.Builder(2).build(), "activation_21")
                // route
                .addVertex("concatenate_21", new MergeVertex(),
                        "rearrange_21", "activation_20");
        addLayers(graphBuilder, 22, "concatenate_21", 3, 1024 + 256, 1024, 0, 0);
        graphBuilder.addLayer("convolution2d_23",new ConvolutionLayer.Builder(1,1)
                                .nIn(1024)
                                .nOut(nBoxes * (5 + numClasses))
                                .weightInit(WeightInit.XAVIER)
                                .stride(1,1)
                                .convolutionMode(ConvolutionMode.Same)
                                .weightInit(WeightInit.RELU)
                                .activation(Activation.IDENTITY)
                                .cudnnAlgoMode(cudnnAlgoMode)
                                .build(),
                        "activation_22")
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .boundingBoxPriors(priors)
                                .build(),
                        "convolution2d_23")
                .setOutputs("outputs");
        return graphBuilder.build();
    }


    /**
    *   @desc : 网络初始化
    *   @auth : TYF
    *   @date : 2019-08-01 - 17:00
    */
    public static ComputationGraph init(){
        ComputationGraph model = new ComputationGraph(conf());
        model.init();
        System.out.println(model.summary(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0])));
        return model;
    }

    /**
    *   @desc : 训练
    *   @auth : TYF
    *   @date : 2019-08-01 - 17:01
    */
    public static void train() throws Exception{

        ComputationGraph yolo2 = init();

        //load数据
        File imageDir = new File(new File("").getCanonicalPath()+"/JPEGImages");
        System.out.println("imageDir:"+imageDir);
        RandomPathFilter pathFilter = new RandomPathFilter(rng) {
            @Override
            protected boolean accept(String name) {
                //按照名称读取pic对应的voc
                name = name.replace("/JPEGImages/", "/Annotations/").replace(".jpg", ".xml");
                try {
                    return new File(new URI(name)).exists();
                } catch (URISyntaxException ex) {
                    throw new RuntimeException(ex);
                }
            }
        };
        //训练集
        InputSplit[] data = new FileSplit(imageDir, NativeImageLoader.ALLOWED_FORMATS, rng).sample(pathFilter);
        InputSplit trainData = data[0];
        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(inputShape[2], inputShape[1], inputShape[0], gridHeight, gridWidth,new VocLabelProvider(new File("").getCanonicalPath()));
        recordReaderTrain.initialize(trainData);
        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        //可视化
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        yolo2.setListeners(new StatsListener(statsStorage));
        yolo2.setListeners(new ScoreIterationListener(1));
        for (int i = 1; i <= nEpochs; i++) {
            train.reset();
            System.out.println("nEpochs:"+i);
            while (train.hasNext()) {
                yolo2.fit(train.next());
            }
        }
        ModelSerializer.writeModel(yolo2, "xxx", true);
        //Runtime.getRuntime().exec("shutdown -s -t 30");
    }


    /**
    *   @desc : 预测
    *   @auth : TYF
    *   @date : 2019-08-09 - 14:15
    */
    public static void predict(){

    }


    public static void main(String[] args) throws Exception{

        train();

    }

}
