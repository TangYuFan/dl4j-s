package nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @desc : nd4j基本操作
 * @auth : TYF
 * @data : 2019/6/13 19:44
 */
public class t_1 {

    //INDArray矩阵创建
    public static void test1(){
        //3行5列 全0矩阵
        INDArray zeros = Nd4j.zeros(3,5);
        System.out.println("zeros:"+zeros);
        //3行5列 全1矩阵
        INDArray ones = Nd4j.ones(3,5);
        System.out.println("ones:"+ones);
        //3行5列 元素随机
        INDArray rands = Nd4j.rand(3,5);
        System.out.println("rands:"+rands);
        //3行5列 元素服从高斯分布(均值为1标准差为0)
        INDArray randns = Nd4j.randn(3,5);
        System.out.println("randns:"+randns);
        //给定1维 转为自定义shape
        INDArray array1 = Nd4j.create(new float[]{2,2,2,2},new int[]{1,4});//一行4列
        System.out.println("array1:"+array1);
    }

    //INDArray矩阵值操作
    public static void test2(){
        //读取值
        INDArray array1 = Nd4j.create(new float[]{1,2,3,4,5,6,7,8,9,10,11,12},new int[]{2,6});//2行6列
        System.out.println("原数组:"+array1);
        System.out.println("下标(0,1):"+array1.getDouble(0,3));
        //修改值
        array1.putScalar(0,3,100);
        System.out.println("下标(0,3):"+array1.getDouble(0,3));
        //获取第0行
        System.out.println("第0行:"+array1.getRow(0));
        //获取第0,1行
        System.out.println("第0,1行:"+array1.getRows(0,1));
        //替换第0行
        array1.putRow(0,Nd4j.create(new float[]{1,2,3,4,5,6}));
        System.out.println("第0行:"+array1.getRow(0));
    }


    //矩阵运算
    public static void test3(){

        //行向量
        INDArray nd1 = Nd4j.create(new float[]{1,2,3,4},new int[]{1,4});//1行4列
        //列向量
        INDArray nd2 = Nd4j.create(new float[]{1,2,3,4},new int[]{4,1});//4行1列
        //方阵
        INDArray nd3 = Nd4j.create(new float[]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},new int[]{4,4});//4行1列
        //打印
        System.out.println("nd1:"+nd1);
        System.out.println("nd2:"+nd2);
        System.out.println("nd3:"+nd3);
        //1行4列 乘 1行1列
        INDArray result1 = nd1.mmul(nd2);
        //1行4列 乘 4行4列
        INDArray result2 = nd1.mmul(nd3);
        //4行4列 乘 4行4列
        INDArray result3 = nd3.mmul(nd3);
        System.out.println("result1:"+result1);
        System.out.println("result2:"+result2);
        System.out.println("result3:"+result3);

    }

    //其他操作
    public static void test4(){
        //https://blog.csdn.net/u011669700/article/details/80139619

        //张量操作
        //加上一个值： arr1.add(myDouble)
        //减去一个值：arr1.sub(myDouble)
        //乘以一个值：arr.mul(myDouble)
        //除以一个值：arr.div(myDouble)
        //减法反操作（scalar - arr1）：arr1.rsub(myDouble)
        //除法反操作（scalar / arr1）：arr1.rdiv(myDouble)

        //元素操作
        //加：arr1.add(arr2)
        //减：arr1.sub(arr2)
        //乘：arr1.mul(arr2)
        //除：arr1.div(arr2)
        //赋值：arr1.assign(arr2)

        //规约操作
        //所有元素的和：arr.sumNumber()
        //所有元素的乘积：arr.prod()
        //L1或者L2范数：arr.norm1() arr.norm2()
        //所有元素的标准差：arr.stdNumber()

        //线性代数操作
        //矩阵乘法：arr1.mmul(arr2)
        //矩阵转置：transpose()
        //获取对角矩阵：Nd4j.diag(INDArray)
        //矩阵求逆：InvertMatrix.invert(INDArray,boolean)

        //元素级变换
        //使用 Transform :Transforms.sin(INDArray) Transforms.log(INDArray) Transforms.sigmoid(INDArray)
        //方法1： Nd4j.getExecutioner().execAndReturn(new Tanh(INDArray))
        //方法2： Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh",INDArray))



    }

    public static void main(String[] args) {

    }
}
