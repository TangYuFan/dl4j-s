package com.example.yolo3coco.controller;

import com.example.yolo3coco.model.graph.BoxPosition;
import com.example.yolo3coco.model.graph.Recognition;
import com.example.yolo3coco.model.view.CocoObject;
import com.example.yolo3coco.util.ObjectDetector;
import org.apache.commons.io.IOUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

@Controller
@RequestMapping("/")
public class CocoController {

    @Autowired
    private ObjectDetector objectDetector;

    @PostMapping("/recognize")
    @ResponseBody
    public List<CocoObject> recognize(@RequestParam("image") MultipartFile image) throws IOException {

        byte[] imageContent = IOUtils.toByteArray(image.getInputStream());

        // 少有人用java做人工智能，因为通过jvm调用CPU的计算效率太低了。
        // 如果使用底层驱动调用GPU计算，可以在近实时时间(小于50毫秒)完成一张图片的识别。
        List<Recognition> objects = objectDetector.detect(imageContent);

        List<CocoObject> result = objects.stream()
                .map(r -> {
                    BoxPosition box = r.getLocation();
                    int probability = Double.valueOf(Math.ceil(r.getConfidence() * 100.0)).intValue();
                    CocoObject cocoObject = new CocoObject(r.getId(), r.getTitle(), probability, box.getLeftInt(), box.getTopInt(), box.getRightInt(), box.getBottomInt());
                    return cocoObject;
                }).collect(Collectors.toList());

        return result;
    }

    @GetMapping("/")
    public String index() {
        return "index";
    }
}
