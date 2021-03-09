package com.example.yolo3coco.util;

import com.example.yolo3coco.model.graph.BoundingBox;
import com.example.yolo3coco.model.graph.BoxPosition;
import com.example.yolo3coco.model.graph.Recognition;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.analysis.function.Sigmoid;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * YOLOClassifier class implemented in Java by using the TensorFlow Java API
 * 参考并魔改自： https://github.com/szaza/android-yolo-v2
 */
public class YOLOClassifier {

    // 下面这些参数需要针对训练出的模型做调整
    private final float OVERLAP_THRESHOLD = 0.5f;
    private double[] SMALL_CELL_ANCHORS;
    private double[] MIDDLE_CELL_ANCHORS;
    private double[] BIG_CELL_ANCHORS;
    private int MAX_RECOGNIZED_CLASSES = 80;
    private final float THRESHOLD = 0.5f;
    private final int MAX_RESULTS = 24;
    private final Integer imageSize = 416;

    public YOLOClassifier(double[] anchors, int recognizeClasses) {
        this.SMALL_CELL_ANCHORS = ArrayUtils.subarray(anchors, 0, 6);
        this.MIDDLE_CELL_ANCHORS = ArrayUtils.subarray(anchors, 6, 12);
        this.BIG_CELL_ANCHORS = ArrayUtils.subarray(anchors, 12, 18);

        MAX_RECOGNIZED_CLASSES = recognizeClasses;
    }

    /**
     * It classifies the object/objects on the image
     *
     * @param tensorFlowOutput output from the TensorFlow, it is a 19x19x(NUMBER_OF_BOUNDING_BOX)x(5 + classes.length) tensor
     * @param labels a string vector with the labels
     * @return a list of recognition objects
     */
    public List<Recognition> classifyImage(final float[][] tensorFlowOutput, final List<String> labels) {

        // 模型中使用的参数
        final int bigCellSplit = 13;
        final int midCellSplit = 26;
        final int smallCellSplit = 52;

        float[] bigCellOutput = tensorFlowOutput[0];
        float[] midCellOutput = tensorFlowOutput[1];
        float[] smallCellOutput = tensorFlowOutput[2];

        BoundingBox[][][] bigCellBoundingBox = new BoundingBox[bigCellSplit][bigCellSplit][3];
        BoundingBox[][][] midCellBoundingBox = new BoundingBox[midCellSplit][midCellSplit][3];
        BoundingBox[][][] smallCellBoundingBox = new BoundingBox[smallCellSplit][smallCellSplit][3];

        PriorityQueue<Recognition> priorityQueue = new PriorityQueue(MAX_RECOGNIZED_CLASSES, new RecognitionComparator());

        fillPriorityQueue(bigCellSplit, bigCellOutput, BIG_CELL_ANCHORS, bigCellBoundingBox, priorityQueue, labels);
        fillPriorityQueue(midCellSplit, midCellOutput, MIDDLE_CELL_ANCHORS, midCellBoundingBox, priorityQueue, labels);
        fillPriorityQueue(smallCellSplit, smallCellOutput, SMALL_CELL_ANCHORS, smallCellBoundingBox, priorityQueue, labels);

        return getRecognition(priorityQueue);
    }

    private void fillPriorityQueue(
            int cellSplit,
            float[] cellOutput,
            double[] cellAnchors,
            BoundingBox[][][] boundingBoxPerCell,
            PriorityQueue<Recognition> priorityQueue,
            List<String> labels) {

        int offset = 0;
        for (int cy=0; cy<cellSplit; cy++) {        // SIZE * SIZE cells
            for (int cx=0; cx<cellSplit; cx++) {
                for (int b=0; b<3; b++) {   // 3 bounding boxes per each cell
                    boundingBoxPerCell[cx][cy][b] = getModel(cellOutput, cx, cy, cellAnchors, b, offset, cellSplit);
                    calculateTopPredictions(boundingBoxPerCell[cx][cy][b], priorityQueue, labels);
                    offset = offset + MAX_RECOGNIZED_CLASSES + 5;
                }
            }
        }
    }


    private BoundingBox getModel(final float[] tensorFlowOutput, int cx, int cy, double[] anchors, int b, int offset, int cellSplit) {

        float x = tensorFlowOutput[offset];
        float y = tensorFlowOutput[offset + 1];
        float width = tensorFlowOutput[offset + 2];
        float height = tensorFlowOutput[offset + 3];
        float confidence = tensorFlowOutput[offset + 4];

        BoundingBox model = new BoundingBox();
        Sigmoid sigmoid = new Sigmoid();
        model.setX((cx + sigmoid.value(x)) / cellSplit);
        model.setY((cy + sigmoid.value(y)) / cellSplit);
        model.setWidth(Math.exp(width) * anchors[2 * b] / imageSize);
        model.setHeight(Math.exp(height) * anchors[2 * b + 1] / imageSize);
        model.setConfidence(sigmoid.value(confidence));

        model.setClasses(new double[MAX_RECOGNIZED_CLASSES]);

        for (int probIndex=0; probIndex<MAX_RECOGNIZED_CLASSES; probIndex++) {
            model.getClasses()[probIndex] = tensorFlowOutput[probIndex + offset + 5];
        }

        return model;
    }

    private void calculateTopPredictions(final BoundingBox boundingBox, final PriorityQueue<Recognition> predictionQueue,
                                         final List<String> labels) {

        ArgMax.Result argMax = new ArgMax(new SoftMax(boundingBox.getClasses()).getValue()).getResult();
        double confidenceInClass = argMax.getMaxValue() * boundingBox.getConfidence();
        if (confidenceInClass > THRESHOLD) {
            predictionQueue.add(new Recognition(argMax.getIndex(), labels.get(argMax.getIndex()), (float) confidenceInClass,
                    new BoxPosition((float) (boundingBox.getX() - boundingBox.getWidth() / 2),
                            (float) (boundingBox.getY() - boundingBox.getHeight() / 2),
                            (float) boundingBox.getWidth(),
                            (float) boundingBox.getHeight())));
        }

    }

    private List<Recognition> getRecognition(final PriorityQueue<Recognition> priorityQueue) {
        List<Recognition> recognitions = new ArrayList();

        if (priorityQueue.size() > 0) {
            // Best recognition
            Recognition bestRecognition = priorityQueue.poll();
            recognitions.add(bestRecognition);

            for (int i = 0; i < Math.min(priorityQueue.size(), MAX_RESULTS); ++i) {
                Recognition recognition = priorityQueue.poll();
                boolean overlaps = false;
                for (Recognition previousRecognition : recognitions) {
                    overlaps = overlaps || (getIntersectionProportion(previousRecognition.getLocation(),
                            recognition.getLocation()) > OVERLAP_THRESHOLD);
                }

                if (!overlaps) {
                    recognitions.add(recognition);
                }
            }
        }

        return recognitions;
    }

    private float getIntersectionProportion(BoxPosition primaryShape, BoxPosition secondaryShape) {
        if (overlaps(primaryShape, secondaryShape)) {
            float intersectionSurface = Math.max(0, Math.min(primaryShape.getRight(), secondaryShape.getRight()) - Math.max(primaryShape.getLeft(), secondaryShape.getLeft())) *
                    Math.max(0, Math.min(primaryShape.getBottom(), secondaryShape.getBottom()) - Math.max(primaryShape.getTop(), secondaryShape.getTop()));

            float surfacePrimary = Math.abs(primaryShape.getRight() - primaryShape.getLeft()) * Math.abs(primaryShape.getBottom() - primaryShape.getTop());

            return intersectionSurface / surfacePrimary;
        }

        return 0f;

    }

    private boolean overlaps(BoxPosition primary, BoxPosition secondary) {
        return primary.getLeft() < secondary.getRight() && primary.getRight() > secondary.getLeft()
                && primary.getTop() < secondary.getBottom() && primary.getBottom() > secondary.getTop();
    }

    // Intentionally reversed to put high confidence at the head of the queue.
    private class RecognitionComparator implements Comparator<Recognition> {
        @Override
        public int compare(final Recognition recognition1, final Recognition recognition2) {
            return Float.compare(recognition2.getConfidence(), recognition1.getConfidence());
        }
    }
}