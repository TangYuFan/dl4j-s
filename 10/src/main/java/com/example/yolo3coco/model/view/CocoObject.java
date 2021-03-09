package com.example.yolo3coco.model.view;

/**
 * 识别出的对象
 */
public class CocoObject {

    public CocoObject() {}

    public CocoObject(Integer id,  String name, Integer probability,  Integer x1, Integer y1, Integer x2, Integer y2) {
        this.id = id;
        this.name = name;
        this.probability = probability;
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
    }

    // 类别编号
    private Integer id;

    // 类别名称
    private String name;

    // 可能性
    private Integer probability;

    // 坐标x1
    private Integer x1;

    // 坐标y1
    private Integer y1;

    // 坐标x2
    private Integer x2;

    // 坐标y2
    private Integer y2;

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Integer getProbability() {
        return probability;
    }

    public void setProbability(Integer probability) {
        this.probability = probability;
    }

    public Integer getX1() {
        return x1;
    }

    public void setX1(Integer x1) {
        this.x1 = x1;
    }

    public Integer getY1() {
        return y1;
    }

    public void setY1(Integer y1) {
        this.y1 = y1;
    }

    public Integer getX2() {
        return x2;
    }

    public void setX2(Integer x2) {
        this.x2 = x2;
    }

    public Integer getY2() {
        return y2;
    }

    public void setY2(Integer y2) {
        this.y2 = y2;
    }
}
