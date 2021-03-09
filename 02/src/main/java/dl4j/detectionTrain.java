package dl4j;


import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.records.metadata.RecordMetaDataImageURI;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Random;
import static org.bytedeco.javacpp.opencv_core.CV_8U;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * @desc : yolo算法目标检测
 * @auth : TYF
 * @data : 2019/6/12
 */
public class detectionTrain {

    private static final Logger log = LoggerFactory.getLogger(detectionTrain.class);

    public static void train() throws Exception {

        //项目根目录
        String path = new File("").getCanonicalPath();

        //yolo基本参数
        int width = 960;
        int height = 540;
        int nChannels = 3;
        int gridWidth = 30;
        int gridHeight = 17;

        //标签数量
        int nClasses = 1;

        //输出层参数
        int nBoxes = 5;
        double lambdaNoObj = 0.5;
        double lambdaCoord = 5.0;
        double[][] priorBoxes = { { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 } };
        double detectionThreshold = 0.3;

        //训练参数
        int batchSize = 2;
        int nEpochs = 50;
        double learningRate = 1e-3;
        double lrMomentum = 0.9;
        int seed = 123;
        Random rng = new Random(seed);

        String dataDir = path;
        File imageDir = new File(path+"/JPEGImages");

        log.info("load data...");
        RandomPathFilter pathFilter = new RandomPathFilter(rng) {
            @Override
            protected boolean accept(String name) {
                //按招名称读取pic对应的voc
                name = name.replace("/JPEGImages/", "/Annotations/").replace(".jpg", ".xml");
                try {
                    return new File(new URI(name)).exists();
                } catch (URISyntaxException ex) {
                    throw new RuntimeException(ex);
                }
            }
        };
        InputSplit[] data = new FileSplit(imageDir, NativeImageLoader.ALLOWED_FORMATS, rng).sample(pathFilter, 0.8, 0.2);
        InputSplit trainData = data[0];
        InputSplit testData = data[1];

        //训练集
        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(height, width, nChannels, gridHeight, gridWidth,new VocLabelProvider(dataDir));
        recordReaderTrain.initialize(trainData);

        //测试集
        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(height, width, nChannels, gridHeight, gridWidth,new VocLabelProvider(dataDir));
        recordReaderTest.initialize(testData);

        //归一化
        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        //下载预训练模型
        ComputationGraph model;
        String modelFilename = path+"/model.zip";
        if (new File(modelFilename).exists()) {
            log.info("load model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);
        } else {
            log.info("create model...");
            //下载预训练模型
            ComputationGraph pretrained = (ComputationGraph)TinyYOLO.builder().build().initPretrained();
            INDArray priors = Nd4j.create(priorBoxes);

            //修改与训练模型结构
            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration
                    .Builder().seed(seed)
                    //优化算法:随机梯度下降
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    //梯度标准化算法:RenormalizeL2PerLayer梯度(防止梯度消失和梯度爆炸)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                    .gradientNormalizationThreshold(1.0)
                    //更新器:Nesterovs
                    .updater(new Adam.Builder().learningRate(learningRate).build())
                    .updater(new Nesterovs.Builder().learningRate(learningRate).momentum(lrMomentum).build())
                    .activation(Activation.IDENTITY)
                    //内存管理模式:工作区
                    .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                    .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                    .build();

            //使用迁移学习对于模型架构记性修改
            model = new TransferLearning
                    .GraphBuilder(pretrained).
                    fineTuneConfiguration(fineTuneConf).
                    removeVertexKeepConnections("conv2d_9")
                    .addLayer("convolution2d_9",new ConvolutionLayer.Builder(1, 1).nIn(1024).nOut(nBoxes * (5 + nClasses)).stride(1, 1).convolutionMode(ConvolutionMode.Same).weightInit(WeightInit.UNIFORM).hasBias(false).activation(Activation.IDENTITY).build(), "leaky_re_lu_8")
                    .addLayer("outputs", new Yolo2OutputLayer.Builder().lambbaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord).boundingBoxPriors(priors).build(),"convolution2d_9")
                    .setOutputs("outputs")
                    .build();

            System.out.println(model.summary(InputType.convolutional(height, width, nChannels)));

            log.info("train...");
            model.setListeners(new ScoreIterationListener(1));
            for (int i = 0; i < nEpochs; i++) {
                train.reset();
                while (train.hasNext()) {
                    model.fit(train.next());
                }
                log.info("*** Completed epoch {} ***", i);
            }
            //保存模型
            ModelSerializer.writeModel(model, modelFilename, true);
        }

        //模型检测可视化
        NativeImageLoader imageLoader = new NativeImageLoader();
        CanvasFrame frame = new CanvasFrame("detectionTrain");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
        List<String> labels = train.getLabels();
        test.setCollectMetaData(true);
        while (test.hasNext() && frame.isVisible()) {
            org.nd4j.linalg.dataset.DataSet ds = test.next();
            RecordMetaDataImageURI metadata = (RecordMetaDataImageURI) ds.getExampleMetaData().get(0);
            INDArray features = ds.getFeatures();
            INDArray results = model.outputSingle(features);
            List<DetectedObject> objs = yout.getPredictedObjects(results, detectionThreshold);
            File file = new File(metadata.getURI());
            log.info(file.getName() + ": " + objs);
            Mat mat = imageLoader.asMat(features);
            Mat convertedMat = new Mat();
            mat.convertTo(convertedMat, CV_8U, 255, 0);
            int w = metadata.getOrigW() * 2;
            int h = metadata.getOrigH() * 2;
            Mat image = new Mat();
            resize(convertedMat, image, new Size(w, h));
            for (DetectedObject obj : objs) {
                double[] xy1 = obj.getTopLeftXY();
                double[] xy2 = obj.getBottomRightXY();
                String label = labels.get(obj.getPredictedClass());
                int x1 = (int) Math.round(w * xy1[0] / gridWidth);
                int y1 = (int) Math.round(h * xy1[1] / gridHeight);
                int x2 = (int) Math.round(w * xy2[0] / gridWidth);
                int y2 = (int) Math.round(h * xy2[1] / gridHeight);
                rectangle(image, new Point(x1, y1), new Point(x2, y2), Scalar.RED);
                putText(image, label, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
            }
            frame.setTitle(new File(metadata.getURI()).getName() + " - detectionTrain");
            frame.setCanvasSize(w, h);
            frame.showImage(converter.convert(image));
            frame.waitKey();
        }
        frame.dispose();
    }

    public static void main(String[] args) throws Exception {
        train();
    }

}