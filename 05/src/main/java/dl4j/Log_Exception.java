package dl4j;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;

/**
 * @desc : 保存日志
 * @auth : TYF
 * @date : 2019-06-26 - 16:38
 */
public class Log_Exception {

    //保存到txt
    public static void writeEror_to_txt(String path,String content) throws IOException{
 
        File F=new File(path);
        //如果文件不存在,就动态创建文件
        if(!F.exists()){
            F.createNewFile();
        }
        FileWriter fw=null;
        String writeDate="时间:"+get_nowDate()+"---"+"error:"+content;
        try {
            //设置为:True,表示写入的时候追加数据
            fw=new FileWriter(F, true);
            //回车并换行
            fw.write(writeDate+"\r\n");
        } catch (IOException e) {
            e.printStackTrace();
        }finally{
            if(fw!=null){
                fw.close();
            }
        }
 
    }
    //获取系统时间
    public static String get_nowDate(){
 
        Calendar D=Calendar.getInstance();
        int year=0;
        int moth=0;
        int day=0;
        year=D.get(Calendar.YEAR);
        moth=D.get(Calendar.MONTH)+1;
        day=D.get(Calendar.DAY_OF_MONTH);
        String now_date=String.valueOf(year)+"-"+String.valueOf(moth)+"-"+String.valueOf(day);
        return now_date;
    }
    //测试方法
    public static void main(String[] args) throws IOException {
        String path="E:/filezl.txt";
        String content = null;
        try{
            List<String> list=new ArrayList<>();
            list.add("1");
            list.add("2");
            list.add("3");
            for(String  i:list){
                System.out.println(i);
            }
            String j=list.get(3);
        }catch (Exception e){
           content=e.getClass().getName()+"  error Info  "+e.getMessage();
        }
        Log_Exception le=new Log_Exception();
        le.writeEror_to_txt(path, content);
    }
}