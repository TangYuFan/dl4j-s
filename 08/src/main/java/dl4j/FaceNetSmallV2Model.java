package dl4j;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.L2NormalizeVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @desc : faceNet网络
 * @auth : TYF
 * @data : 2019/6/30
 */
public class FaceNetSmallV2Model {

	private final static Logger LOGGER = LoggerFactory.getLogger(FaceNetSmallV2Model.class);

    private final long seed = 1234;
    private int[] inputShape = new int[]{3, 96, 96};
    private IUpdater updater = new Adam(0.1, 0.9, 0.999, 0.01);
    private int encodings = 128;
    public static int reluIndex = 1;
    public static int paddingIndex = 1;

    public ComputationGraphConfiguration conf() {
    	LOGGER.info("conf............");
        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .activation(Activation.IDENTITY)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(updater)
                .weightInit(WeightInit.RELU)
                .l2(5e-5)
                .miniBatch(true)
                .graphBuilder();

        graph.addInputs("input1")
                .addLayer("pad1",
                        zeroPadding(3), "input1")
                .addLayer("conv1",
                        convolution(7, inputShape[0], 64, 2),
                        "pad1")
                .addLayer("bn1", batchNorm(64),
                        "conv1")
                .addLayer(nextReluId(), relu(),
                        "bn1")
                .addLayer("pad2",
                        zeroPadding(1), lastReluId())
                // pool -> norm
                .addLayer("pool1",
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3},
                                new int[]{2, 2})
                                .convolutionMode(ConvolutionMode.Truncate)
                                .build(),
                        "pad2")

                // Inception 2
                .addLayer("conv2",
                        convolution(1, 64, 64),
                        "pool1")
                .addLayer("bn2", batchNorm(64),
                        "conv2")
                .addLayer(nextReluId(),
                        relu(),
                        "bn2")

                .addLayer("pad3",
                        zeroPadding(1), lastReluId())

                .addLayer("conv3",
                        convolution(3, 64, 192),
                        "pad3")
                .addLayer("bn3",
                        batchNorm(192),
                        "conv3")
                .addLayer(nextReluId(),
                        relu(),
                        "bn3")

                .addLayer("pad4",
                        zeroPadding(1), lastReluId())
                .addLayer("pool2",
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3},
                                new int[]{2, 2})
                                .convolutionMode(ConvolutionMode.Truncate)
                                .build(),
                        "pad4");


        buildBlock3a(graph);
        buildBlock3b(graph);
        buildBlock3c(graph);

        buildBlock4a(graph);
        buildBlock4e(graph);

        buildBlock5a(graph);
        buildBlock5b(graph);

        graph.addLayer("avgpool",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{3, 3},
                        new int[]{1, 1})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),
                "inception_5b")
                .addLayer("dense", new DenseLayer.Builder().nIn(736).nOut(encodings)
                        .activation(Activation.IDENTITY).build(), "avgpool")
                .addVertex("encodings", new L2NormalizeVertex(new int[]{}, 1e-12), "dense")
                .setInputTypes(InputType.convolutional(96, 96, inputShape[0])).pretrain(true);
        graph.setOutputs("encodings");

        return graph.build();
    }

    private void buildBlock3a(ComputationGraphConfiguration.GraphBuilder graph) {
        graph.addLayer("inception_3a_3x3_conv1", convolution(1, 192, 96),
                "pool2")
                .addLayer("inception_3a_3x3_bn1",
                        batchNorm(96), "inception_3a_3x3_conv1")
                .addLayer(nextReluId(),
                        relu(), "inception_3a_3x3_bn1")
                .addLayer(nextPaddingId(),
                        zeroPadding(1), lastReluId())
                .addLayer("inception_3a_3x3_conv2", convolution(3, 96, 128), lastPaddingId())
                .addLayer("inception_3a_3x3_bn2",
                        batchNorm(128),
                        "inception_3a_3x3_conv2")
                .addLayer(nextReluId(),
                        relu(), "inception_3a_3x3_bn2")

                .addLayer("inception_3a_5x5_conv1", convolution(1, 192, 16),
                        "pool2")
                .addLayer("inception_3a_5x5_bn1",
                        batchNorm(16),
                        "inception_3a_5x5_conv1")
                .addLayer(nextReluId(),
                        relu(), "inception_3a_5x5_bn1")
                .addLayer(nextPaddingId(),
                        zeroPadding(2), lastReluId())
                .addLayer("inception_3a_5x5_conv2", convolution(5, 16, 32), lastPaddingId())
                .addLayer("inception_3a_5x5_bn2",
                        batchNorm(32),
                        "inception_3a_5x5_conv2")
                .addLayer(nextReluId(),
                        relu(), "inception_3a_5x5_bn2")

                .addLayer("pool3",
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3},
                                new int[]{2, 2})
                                .convolutionMode(ConvolutionMode.Truncate)
                                .build(),
                        "pool2")
                .addLayer("inception_3a_pool_conv", convolution(1, 192, 32), "pool3")
                .addLayer("inception_3a_pool_bn",
                        batchNorm(32),
                        "inception_3a_pool_conv")
                .addLayer(nextReluId(),
                        relu(),
                        "inception_3a_pool_bn")

                .addLayer(nextPaddingId(),
                        new ZeroPaddingLayer.Builder(new int[]{3, 4, 3, 4})
                                .build(), lastReluId())

                .addLayer("inception_3a_1x1_conv", convolution(1, 192, 64),
                        "pool2")
                .addLayer("inception_3a_1x1_bn",
                        batchNorm(64),
                        "inception_3a_1x1_conv")
                .addLayer(nextReluId(),
                        relu(),
                        "inception_3a_1x1_bn")
                .addVertex("inception_3a", new MergeVertex(), "relu5", "relu7", lastPaddingId(), "relu9");

    }


    private void buildBlock3b(ComputationGraphConfiguration.GraphBuilder graph) {
        graph.addLayer("inception_3b_3x3_conv1",
                convolution(1, 256, 96),
                "inception_3a")

                .addLayer("inception_3b_3x3_bn1",
                        batchNorm(96),
                        "inception_3b_3x3_conv1")

                .addLayer(nextReluId(),
                        relu(),
                        "inception_3b_3x3_bn1")

                .addLayer(nextPaddingId(),
                        zeroPadding(1), lastReluId())

                .addLayer("inception_3b_3x3_conv2",
                        convolution(3, 96, 128),
                        lastPaddingId())

                .addLayer("inception_3b_3x3_bn2",
                        batchNorm(128),
                        "inception_3b_3x3_conv2")

                .addLayer(nextReluId(),
                        relu(),
                        "inception_3b_3x3_bn2");


        graph.addLayer("inception_3b_5x5_conv1",
                convolution(1, 256, 32),
                "inception_3a")

                .addLayer("inception_3b_5x5_bn1",
                        batchNorm(32),
                        "inception_3b_5x5_conv1")

                .addLayer(nextReluId(),
                        relu(),
                        "inception_3b_5x5_bn1")
                .addLayer(nextPaddingId(),
                        zeroPadding(2), lastReluId())

                .addLayer("inception_3b_5x5_conv2",
                        convolution(5, 32, 64),
                        lastPaddingId())

                .addLayer("inception_3b_5x5_bn2",
                        batchNorm(64),
                        "inception_3b_5x5_conv2")
                .addLayer(nextReluId(),
                        relu(),
                        "inception_3b_5x5_bn2");

        graph.addLayer("avg1",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{3, 3},
                        new int[]{3, 3})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),
                "inception_3a")
                .addLayer("inception_3b_pool_conv",
                        convolution(1, 256, 64),
                        "avg1")

                .addLayer("inception_3b_pool_bn",
                        batchNorm(64),
                        "inception_3b_pool_conv")

                .addLayer(nextReluId(),
                        relu(),
                        "inception_3b_pool_bn")
                .addLayer(nextPaddingId(),
                        zeroPadding(4), lastReluId())

                .addLayer("inception_3b_1x1_conv",
                        convolution(1, 256, 64),
                        "inception_3a")
                .addLayer("inception_3b_1x1_bn",
                        batchNorm(64),
                        "inception_3b_1x1_conv")

                .addLayer(nextReluId(),
                        relu(),
                        "inception_3b_1x1_bn")
                .addVertex("inception_3b", new MergeVertex(), "relu11", "relu13", lastPaddingId(), "relu15");

    }

    private void buildBlock3c(ComputationGraphConfiguration.GraphBuilder graph) {
        convolution2dAndBN(graph, "inception_3c_3x3",
                128, 320, new int[]{1, 1}, new int[]{1, 1},
                256, 128, new int[]{3, 3}, new int[]{2, 2},
                new int[]{1, 1, 1, 1}, "inception_3b");
        String rel1 = lastReluId();

        convolution2dAndBN(graph, "inception_3c_5x5",
                32, 320, new int[]{1, 1}, new int[]{1, 1},
                64, 32, new int[]{5, 5}, new int[]{2, 2},
                new int[]{2, 2, 2, 2}, "inception_3b");
        String rel2 = lastReluId();

        graph.addLayer("pool7",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3},
                        new int[]{2, 2})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),
                "inception_3b");

        graph.addLayer(nextPaddingId(),
                new ZeroPaddingLayer.Builder(new int[]{0, 1, 0, 1})
                        .build(), "pool7");
        String pad1 = lastPaddingId();

        graph.addVertex("inception_3c", new MergeVertex(), rel1, rel2, pad1);
    }

    private void buildBlock4a(ComputationGraphConfiguration.GraphBuilder graph) {
        convolution2dAndBN(graph, "inception_4a_3x3",
                96, 640, new int[]{1, 1}, new int[]{1, 1},
                192, 96, new int[]{3, 3}, new int[]{1, 1}
                , new int[]{1, 1, 1, 1}, "inception_3c");
        String rel1 = lastReluId();

        convolution2dAndBN(graph, "inception_4a_5x5",
                32, 640, new int[]{1, 1}, new int[]{1, 1},
                64, 32, new int[]{5, 5}, new int[]{1, 1}
                , new int[]{2, 2, 2, 2}, "inception_3c");
        String rel2 = lastReluId();

        graph.addLayer("avg7",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{3, 3},
                        new int[]{3, 3})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),
                "inception_3c");
        convolution2dAndBN(graph, "inception_4a_pool",
                128, 640, new int[]{1, 1}, new int[]{1, 1},
                null, null, null, null
                , new int[]{2, 2, 2, 2}, "avg7");
        String pad1 = lastPaddingId();

        convolution2dAndBN(graph, "inception_4a_1x1",
                256, 640, new int[]{1, 1}, new int[]{1, 1},
                null, null, null, null
                , null, "inception_3c");
        String rel4 = lastReluId();
        graph.addVertex("inception_4a", new MergeVertex(), rel1, rel2, rel4, pad1);

    }

    private void buildBlock4e(ComputationGraphConfiguration.GraphBuilder graph) {
        convolution2dAndBN(graph, "inception_4e_3x3",
                160, 640, new int[]{1, 1}, new int[]{1, 1},
                256, 160, new int[]{3, 3}, new int[]{2, 2},
                new int[]{1, 1, 1, 1}, "inception_4a");
        String rel1 = lastReluId();

        convolution2dAndBN(graph, "inception_4e_5x5",
                64, 640, new int[]{1, 1}, new int[]{1, 1},
                128, 64, new int[]{5, 5}, new int[]{2, 2},
                new int[]{2, 2, 2, 2}, "inception_4a");
        String rel2 = lastReluId();

        graph.addLayer("pool8",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3},
                        new int[]{2, 2})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),
                "inception_4a");
        graph.addLayer(nextPaddingId(),
                new ZeroPaddingLayer.Builder(new int[]{0, 1, 0, 1})
                        .build(), "pool8");
        String pad1 = lastPaddingId();

        graph.addVertex("inception_4e", new MergeVertex(), rel1, rel2, pad1);
    }

    private void buildBlock5a(ComputationGraphConfiguration.GraphBuilder graph) {
        convolution2dAndBN(graph, "inception_5a_3x3",
                96, 1024, new int[]{1, 1}, new int[]{1, 1},
                384, 96, new int[]{3, 3}, new int[]{1, 1},
                new int[]{1, 1, 1, 1}, "inception_4e");
        String relu1 = lastReluId();

        graph.addLayer("avg9",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{3, 3},
                        new int[]{3, 3})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),
                "inception_4e");
        convolution2dAndBN(graph, "inception_5a_pool",
                96, 1024, new int[]{1, 1}, new int[]{1, 1},
                null, null, null, null,
                new int[]{1, 1, 1, 1}, "avg9");
        String pad1 = lastPaddingId();

        convolution2dAndBN(graph, "inception_5a_1x1",
                256, 1024, new int[]{1, 1}, new int[]{1, 1},
                null, null, null, null,
                null, "inception_4e");
        String rel3 = lastReluId();

        graph.addVertex("inception_5a", new MergeVertex(), relu1, pad1, rel3);
    }

    private void buildBlock5b(ComputationGraphConfiguration.GraphBuilder graph) {
        convolution2dAndBN(graph, "inception_5b_3x3",
                96, 736, new int[]{1, 1}, new int[]{1, 1},
                384, 96, new int[]{3, 3}, new int[]{1, 1},
                new int[]{1, 1, 1, 1}, "inception_5a");
        String rel1 = lastReluId();

        graph.addLayer("max2",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3},
                        new int[]{2, 2})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),
                "inception_5a");
        convolution2dAndBN(graph, "inception_5b_pool",
                96, 736, new int[]{1, 1}, new int[]{1, 1},
                null, null, null, null,
                null, "max2");
        graph.addLayer(nextPaddingId(),
                zeroPadding(1), lastReluId());
        String pad1 = lastPaddingId();

        convolution2dAndBN(graph, "inception_5b_1x1",
                256, 736, new int[]{1, 1}, new int[]{1, 1},
                null, null, null, null,
                null, "inception_5a");
        String rel2 = lastReluId();

        graph.addVertex("inception_5b", new MergeVertex(), rel1, pad1, rel2);
    }

    public ComputationGraph init() throws IOException {
        resetIndexes();
        ComputationGraph computationGraph = new ComputationGraph(conf());
        //net初始化
        computationGraph.init();
        //加载权重
        loadWeights(computationGraph);
        return computationGraph;
    }

    private static void resetIndexes() {
        reluIndex = 1;
        paddingIndex = 1;
    }
    static final String BASE = "src/main/resources/weigth/";

    static ActivationLayer relu() {
        return new ActivationLayer.Builder().activation(Activation.RELU).build();
    }

    static ZeroPaddingLayer zeroPadding(int i) {
        return new ZeroPaddingLayer.Builder(new int[]{i, i, i, i})
                .build();
    }

    static ConvolutionLayer convolution(int filterSize, int in, int out) {
        return new ConvolutionLayer.Builder(new int[]{filterSize, filterSize})
                .convolutionMode(ConvolutionMode.Truncate)
                .nIn(in).nOut(out)
                .build();
    }

    static ConvolutionLayer convolution(int filterSize, int in, int out, int strides) {
        return new ConvolutionLayer.Builder(new int[]{filterSize, filterSize}, new int[]{strides, strides})
                .convolutionMode(ConvolutionMode.Truncate)
                .nIn(in).nOut(out)
                .build();
    }

    static void convolution2dAndBN(ComputationGraphConfiguration.GraphBuilder graph, String layerName,
                                   Integer conv1Out, Integer conv1In, int[] conv1Filter, int[] conv1Strides,
                                   Integer conv2Out, Integer conv2in, int[] conv2Filter, int[] conv2Strides,
                                   int[] padding, String lastLayer) {

        String num = (conv2Out == null) ? "" : "1";

        graph.addLayer(layerName + "_conv" + num,
                new ConvolutionLayer.Builder(conv1Filter, conv1Strides).nIn(conv1In).nOut(conv1Out)
                        .convolutionMode(ConvolutionMode.Truncate).build(), lastLayer)

                .addLayer(layerName + "_bn" + num,
                        batchNorm(conv1Out),
                        layerName + "_conv" + num)

                .addLayer(nextReluId(),
                        relu(),
                        layerName + "_bn" + num);

        if (padding == null) {
            return;
        }
        graph.addLayer(nextPaddingId(),
                new ZeroPaddingLayer.Builder(padding)
                        .build(), lastReluId());
        if (conv2Out == null) {
            return;
        }
        graph.addLayer(layerName + "_conv2",
                new ConvolutionLayer.Builder(conv2Filter, conv2Strides).nIn(conv2in).nOut(conv2Out)
                        .convolutionMode(ConvolutionMode.Truncate).build(),
                lastPaddingId())

                .addLayer(layerName + "_bn2",
                        batchNorm(conv2Out),
                        layerName + "_conv2")

                .addLayer(nextReluId(),
                        relu(),
                        layerName + "_bn2");

    }

    static BatchNormalization batchNorm(int in) {
        return new BatchNormalization.Builder(false).eps(0.00001).nIn(in).nOut(in).build();
    }

    static String nextReluId() {
        return "relu" + reluIndex++;
    }

    static String nextPaddingId() {
        return "padding" + (paddingIndex++);
    }

    static String lastPaddingId() {
        return "padding" + (paddingIndex - 1);
    }

    static String lastReluId() {
        return "relu" + (reluIndex - 1);
    }

    static double[] readWightsValues(String path) throws IOException {
        String collect = Files.lines(Paths.get(path))
                .collect(Collectors.joining(","));
        return Arrays.stream(collect.split(",")).mapToDouble(Double::parseDouble).toArray();
    }

    static void loadWeights(ComputationGraph computationGraph) throws IOException {

        Layer[] layers = computationGraph.getLayers();
        for (Layer layer : layers) {
            List<double[]> all = new ArrayList<>();
            String layerName = layer.conf().getLayer().getLayerName();
            if (layerName.contains("bn")) {
                all.add(readWightsValues(BASE + layerName + "_w.csv"));
                all.add(readWightsValues(BASE + layerName + "_b.csv"));
                all.add(readWightsValues(BASE + layerName + "_m.csv"));
                all.add(readWightsValues(BASE + layerName + "_v.csv"));
                layer.setParams(mergeAll(all));
            } else if (layerName.contains("conv")) {
                all.add(readWightsValues(BASE + layerName + "_b.csv"));
                all.add(readWightsValues(BASE + layerName + "_w.csv"));
                layer.setParams(mergeAll(all));
            } else if (layerName.contains("dense")) {
                double[] w = readWightsValues(BASE + layerName + "_w.csv");
                all.add(w);
                double[] b = readWightsValues(BASE + layerName + "_b.csv");
                all.add(b);
                layer.setParams(mergeAll(all));
            }
        }
    }

    private static INDArray mergeAll(List<double[]> all) {
        LOGGER.info("mergeAll........................");
        INDArray[] allArr = new INDArray[all.size()];
        int index = 0;
        for (double[] doubles : all) {
            allArr[index++] = Nd4j.create(doubles);
        }
        return Nd4j.toFlattened(allArr);
    }


}
