package com.example.yolo3coco.util;

import com.example.yolo3coco.model.graph.BoxPosition;
import com.example.yolo3coco.model.graph.Recognition;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.List;

/**
 * 参考并魔改自： https://github.com/szaza/android-yolo-v2
 */
public class ObjectDetector {

    private byte[] GRAPH_DEF;
    private List<String> LABELS;
    private YOLOClassifier YOLO_CLASSIFIER;
    private final Integer imageSize = 416;

    public ObjectDetector(byte[] graphDef, List<String> labels, YOLOClassifier yoloClassifier) {
        GRAPH_DEF = graphDef;
        LABELS = labels;
        YOLO_CLASSIFIER = yoloClassifier;
    }

    /**
     * Detect objects on the given image
     */
    public List<Recognition> detect(byte[] imageContent) throws IOException {

        // 把图片转换成容易识别的尺寸和格式
        BufferedImage image = ImageIO.read(new ByteArrayInputStream(imageContent));
        int imageWidth = image.getWidth();
        int imageHeight = image.getHeight();
        double scale = imageSize * 1.0 / Math.max(imageWidth, imageHeight);
        int newWidth = Double.valueOf(imageWidth * scale).intValue();
        int newHeight = Double.valueOf(imageHeight * scale).intValue();
        int addNewWidth = imageSize - newWidth;
        int addNewHeight = imageSize - newHeight;
        Image scaledImage = image.getScaledInstance(newWidth, newHeight, Image.SCALE_SMOOTH);
        BufferedImage newImage = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D imageGraph = (Graphics2D) newImage.getGraphics();
        imageGraph.drawImage(scaledImage, (imageSize - newWidth) / 2, (imageSize - newHeight) / 2, null);
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        ImageIO.write(newImage, "jpeg", outputStream);
        byte[] newImageContent = outputStream.toByteArray();

        // 识别图片内容
        try (Tensor<Float> imageTensor = vectorizeImage(newImageContent)) {
            float[][] imageOutput = executeYOLOGraph(imageTensor);
            List<Recognition> recognitions = YOLO_CLASSIFIER.classifyImage(imageOutput, LABELS);

            // 取出并还原识别的坐标
            for (Recognition recognition : recognitions) {
                BoxPosition box = recognition.getLocation();
                box = new BoxPosition(box, imageSize.floatValue(), imageSize.floatValue());
                float realLeft = box.getLeft() - addNewWidth / 2;
                float realTop = box.getTop() - addNewHeight / 2;
                box = new BoxPosition(realLeft, realTop, box.getWidth(), box.getHeight());
                box = new BoxPosition(box, Double.valueOf(1 / scale).floatValue(), Double.valueOf(1 / scale).floatValue());
                recognition.setLocation(box);
            }
            return recognitions;
        }
    }



    /**
     * Pre-process input. It vectorize the image and normalize its pixels
     */
    private Tensor<Float> vectorizeImage(final byte[] imageBytes) {
        try (Graph graph = new Graph()) {
            GraphBuilder graphBuilder = new GraphBuilder(graph);

            final Output<Float> output =
                    graphBuilder.div( // Divide each pixels with the MEAN
                            graphBuilder.expandDims( // Increase the output tensors dimension
                                    graphBuilder.cast( // Cast the output to Float
                                            graphBuilder.decodeJpeg(
                                                    graphBuilder.constant("input", imageBytes), 3),
                                            Float.class),
                                    graphBuilder.constant("make_batch", 0)),
                            graphBuilder.constant("scale", 255F));

            try (Session session = new Session(graph)) {
                return session.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
            }
        }
    }



    /**
     * Executes graph on the given preprocessed image
     *
     * @param image preprocessed image
     * @return output tensor returned by tensorFlow
     */
    private float[][] executeYOLOGraph(final Tensor<Float> image) {
        try (Graph graph = new Graph()) {
            graph.importGraphDef(GRAPH_DEF);
            try (Session session = new Session(graph)) {
                List<Tensor<?>> outputs = session.runner()
                        .feed("input_1", image)
                        .fetch("output_node0")
                        .fetch("output_node1")
                        .fetch("output_node2")
                        .run();

                Tensor<Float> bigCellOutput = outputs.get(0).expect(Float.class);
                Tensor<Float> midCellOutput = outputs.get(1).expect(Float.class);
                Tensor<Float> smallCellOutput = outputs.get(2).expect(Float.class);

                // 模型中固定的参数
                float[] output1 = new float[13 * 13 * 255];
                float[] output2 = new float[26 * 26 * 255];
                float[] output3 = new float[52 * 52 * 255];
                FloatBuffer floatBuffer1 = FloatBuffer.wrap(output1);
                FloatBuffer floatBuffer2 = FloatBuffer.wrap(output2);
                FloatBuffer floatBuffer3 = FloatBuffer.wrap(output3);
                bigCellOutput.writeTo(floatBuffer1);
                midCellOutput.writeTo(floatBuffer2);
                smallCellOutput.writeTo(floatBuffer3);

                bigCellOutput.close();
                midCellOutput.close();
                smallCellOutput.close();

                float[][] result = new float[][] {output1, output2, output3};
                return result;
            }
        }
    }
}