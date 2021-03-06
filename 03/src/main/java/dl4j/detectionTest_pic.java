package dl4j;


import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.api.io.filters.RandomPathFilter;
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
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
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
 * @desc : ????????????????????????
 * @auth : TYF
 * @data : 2019/6/12
 */
public class detectionTest_pic {

    private static final Logger log = LoggerFactory.getLogger(detectionTrain.class);

    public static void train() throws Exception {

        //???????????????
        String path = new File("").getCanonicalPath();

        //yolo????????????
        int width = 960;
        int height = 540;
        int nChannels = 3;
        int gridWidth = 30;
        int gridHeight = 17;

        //????????????
        int nClasses = 1;

        //???????????????
        int nBoxes = 5;
        double lambdaNoObj = 0.5;
        double lambdaCoord = 5.0;
        double[][] priorBoxes = { { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 } };
        double detectionThreshold = 0.3;

        //????????????
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
                //??????????????????pic?????????voc
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

        //?????????
        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(height, width, nChannels, gridHeight, gridWidth,new VocLabelProvider(dataDir));
        recordReaderTrain.initialize(trainData);

        //?????????
        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(height, width, nChannels, gridHeight, gridWidth,new VocLabelProvider(dataDir));
        recordReaderTest.initialize(testData);

        //?????????
        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        //?????????????????????
        ComputationGraph model;
        String modelFilename = path+"/model.zip";
        if (new File(modelFilename).exists()) {
            log.info("load model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);
        } else {
            log.info("create model...");
            //???????????????
            ComputationGraph pretrained = (ComputationGraph)TinyYOLO.builder().build().initPretrained();
            INDArray priors = Nd4j.create(priorBoxes);

            //???????????????????????????
            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration
                    .Builder().seed(seed)
                    //????????????:??????????????????
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    //?????????????????????:RenormalizeL2PerLayer??????(?????????????????????????????????)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                    .gradientNormalizationThreshold(1.0)
                    //?????????:Nesterovs
                    .updater(new Adam.Builder().learningRate(learningRate).build())
                    .updater(new Nesterovs.Builder().learningRate(learningRate).momentum(lrMomentum).build())
                    .activation(Activation.IDENTITY)
                    //??????????????????:?????????
                    .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                    .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                    .build();

            //????????????
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
            //????????????
            ModelSerializer.writeModel(model, modelFilename, true);
            //??????
            Runtime.getRuntime().exec("shutdown -s -t 30");
        }

        //??????????????????
        NativeImageLoader imageLoader = new NativeImageLoader();
        CanvasFrame canvasFrame = new CanvasFrame("????????????");
        NativeImageLoader loader = new NativeImageLoader(height, width, 3);
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        INDArray in = loader.asMatrix(new File(path+"/JPEG/2.jpg"));
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(in);
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
        INDArray out = model.outputSingle(in);
        List<String> labels = train.getLabels();
        //??????????????????????????????
        List<DetectedObject> objs = yout.getPredictedObjects(out, detectionThreshold);
        int w = width;
        int h = height;
        opencv_core.Mat image = new opencv_core.Mat();
        opencv_core.Mat convertedMat = new opencv_core.Mat();
        opencv_core.Mat mat = imageLoader.asMat(in);
        mat.convertTo(convertedMat, CV_8U, 255, 0);
        resize(convertedMat, image, new opencv_core.Size(w, h));
        for (DetectedObject obj : objs) {
            //?????????
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            //??????
            String label = labels.get(obj.getPredictedClass());
            int x1 = (int) Math.round(w * xy1[0] / gridWidth);
            int y1 = (int) Math.round(h * xy1[1] / gridHeight);
            int x2 = (int) Math.round(w * xy2[0] / gridWidth);
            int y2 = (int) Math.round(h * xy2[1] / gridHeight);
            //??????
            rectangle(image, new opencv_core.Point(x1, y1), new opencv_core.Point(x2, y2), opencv_core.Scalar.RED);
            //?????????
            putText(image, label, new opencv_core.Point(x1-80, y2+30), FONT_HERSHEY_DUPLEX, 1, opencv_core.Scalar.RED);
        }
        canvasFrame.setCanvasSize(w, h);
        canvasFrame.showImage(converter.convert(image));

    }

    public static void main(String[] args) throws Exception {
        train();
    }

}