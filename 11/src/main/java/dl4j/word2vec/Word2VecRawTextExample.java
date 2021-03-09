package dl4j.word2vec;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.util.Arrays;
import java.util.Collection;

/**
*   @desc : 用word2vertor获取词向量
*   @auth : TYF
*   @date : 2019-08-21 - 16:03
*/
public class Word2VecRawTextExample {

    private static Logger log = LoggerFactory.getLogger(Word2VecRawTextExample.class);
    private static String textPath = "C:\\Users\\pc\\Desktop\\love.txt" ;
    private static String vertorOutputPath = "C:\\Users\\pc\\Desktop\\word2vec.txt";
    private static String wightOutputPath = "C:\\Users\\pc\\Desktop\\wight.txt";
    public static void main(String[] args) throws Exception {
        //输入文本
        File inputTxt = new File(textPath);
        log.info("加载数据...."+inputTxt.getName());
        SentenceIterator iter = new LineSentenceIterator(inputTxt);
        //每行拆分
        TokenizerFactory token = new DefaultTokenizerFactory();
        //去除特殊符号
        token.setTokenPreProcessor(new CommonPreprocessor());
        log.info("train....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)//词在语料中必须出现的最少次数
                .iterations(1)
                .layerSize(6)  //词向量维度 也是隐藏层的宽度
                .seed(42)
                .windowSize(5) //窗口大小
                .iterate(iter)
                .tokenizerFactory(token)
                .build();
        vec.fit();
        //保存词向量
        WordVectorSerializer.writeWordVectors(vec, vertorOutputPath);
        //保存权重
        WordVectorSerializer.writeWord2VecModel(vec, wightOutputPath);
        //获取相似的10个词
        Collection<String> lst = vec.wordsNearest("love", 10);
        log.info("love相似的10个词语:"+lst);
        //获取某词对应的向量
        double[] wordVector = vec.getWordVector("love");
        log.info("love的词向量:"+(Arrays.toString(wordVector)));
        //余弦相似度
        log.info("love,will余弦相似度:"+vec.similarity("love","will")+"");

        //如果有了新的语料库(没有添加新词)如下进行权重更新
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(wightOutputPath);
        SentenceIterator iterator = new BasicLineIterator(textPath);
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        word2Vec.setTokenizerFactory(tokenizerFactory);
        word2Vec.setSentenceIterator(iterator);
        word2Vec.fit();//用恢复的模型训练是在原有权重上进行权重更新
        WordVectorSerializer.writeWord2VecModel(vec, wightOutputPath);//保存
        log.info("love的词向量:"+(Arrays.toString(word2Vec.getWordVector("love"))));
        log.info("love,will余弦相似度:"+word2Vec.similarity("love","will")+"");
    }
}
