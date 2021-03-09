package com.example.yolo3coco.config;

import com.example.yolo3coco.util.ObjectDetector;
import com.example.yolo3coco.util.YOLOClassifier;
import org.apache.commons.io.IOUtils;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.stereotype.Component;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 配置类
 */
@Component
public class CocoConfig {

    /**
     * 读取训练类别
     */
    @Bean(name = "cocoClasses")
    public String[] getCocoClasses() throws IOException {

        String classContent = IOUtils.resourceToString(
                "model/coco_classes.txt",
                Charset.forName("UTF-8"),
                CocoConfig.class.getClassLoader());
        String[] classes = classContent.split("\n");
        return classes;
    }

    /**
     * 构造分类器
     */
    @Bean(name = "yoloClassifier")
    public YOLOClassifier getYOLOClassifier(
            @Qualifier("cocoClasses") String[] cocoClasses) throws IOException {

        String anchorContent = IOUtils.resourceToString(
                "model/coco_anchors.txt",
                Charset.forName("UTF-8"),
                CocoConfig.class.getClassLoader());
        List<Double> anchorList = Arrays.stream(anchorContent.split(","))
                .map(c -> Double.parseDouble(c.trim()))
                .collect(Collectors.toList());
        double[] anchors = new double[anchorList.size()];
        for(int i = 0; i < anchorList.size() ; i++) {
            anchors[i] = anchorList.get(i);
        }

        YOLOClassifier classifier = new YOLOClassifier(anchors, cocoClasses.length);
        return classifier;
    }

    /**
     * 构造对象探测器
     */
    @Bean(name = "objectDetector")
    public ObjectDetector getObjectDetector(
            @Qualifier("cocoClasses") String[] cocoClasses,
            @Qualifier("yoloClassifier") YOLOClassifier yoloClassifier
    ) throws IOException {
        byte[] modelContent = IOUtils.resourceToByteArray(
                "model/coco_model.pb",
                CocoConfig.class.getClassLoader());
        ObjectDetector detector = new ObjectDetector(modelContent, Arrays.asList(cocoClasses), yoloClassifier);
        return detector;
        // 如果想找从图片对象识别的具体算法，你可能要失望了，java代码到此为止了。
        // 识别算法在coco_model.pb文件里，这个文件不是程序员手写的，而是给计算机一堆图片教它自己练会的。
        // coco_model.pb里是一层又一层的神经网络，有点儿像动物的大脑。想要有识别其它物品能力记忆的神经网络，可以找：13355385397。
    }
}
