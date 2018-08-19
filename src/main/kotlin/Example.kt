import org.deeplearning4j.models.glove.Glove
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator
import org.deeplearning4j.text.sentenceiterator.SentenceIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.io.ClassPathResource

fun configureAndTrainModel(iter: SentenceIterator): Glove {
    val t = DefaultTokenizerFactory()
    t.tokenPreProcessor = CommonPreprocessor()
    println("Building model....")
    Nd4j.setDataType(DataBuffer.Type.DOUBLE)
    val vec = Glove.Builder().iterate(iter)
            .tokenizerFactory(t)
            .alpha(0.75)
            .learningRate(0.1)
            // number of epochs for training
            .epochs(128)
            // cutoff for weighting function
            .xMax(100.0)
            // training is done in batches taken from training corpus
            .batchSize(256)
            // if set to true, batches will be shuffled before training
            .shuffle(false)
            .layerSize(24)
            // if set to true word pairs will be built in both directions, LTR and RTL
            .symmetric(false)
            .maxMemory(1)
            .minLearningRate(0.01)
            .seed(15)
            .windowSize(1)
            .build();

    println(vec.configuration)
    println("Fitting Word2Vec model....")
    vec.fit()
    return vec
}

fun main(args: Array<String>) {
    val data = FileSentenceIterator(ClassPathResource("data-glove0.tmp").file)
    val model = configureAndTrainModel(data)
}