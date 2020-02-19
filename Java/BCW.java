package org.deeplearning4j.examples.dataexamples;

import org.bytedeco.javacv.FrameFilter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.*;
import java.util.*;

/**
 Author :Ankit Grover
 The following program is used to get predictions for a patient as to whether he has BreastCancer or not
 */
public class BreastCancer {
    public static String dataLocalPath; //Stores the data's local path
   public static  File locationToSave;//Path to save the model
    private static int labelIndex;
    private static int numClasses;     //Number of predictive classes
    private static int batchSize;       //2^n
    private static DataSet train;              //Test and training data
    private static DataSet test;
   public static  MultiLayerNetwork model;

    static double learningRate = 0.01;
    static int nEpochs ;

    static int numInputs ;
    static int numOutputs ;
    static int numHiddenNodes ;
    private static  MultiLayerConfiguration conf;

    private static Logger log = LoggerFactory.getLogger(CSVExample.class);//For checking te LOGS

    public static void main(String args[]) throws Exception {

      /*  System.out.println("Hello");
        int seed = 0;
        double learningRate = 0.01;
        int batchSize =32;
        int nEpochs = 10;

        int numInputs = 8;
        int numOutputs = 8;
        int numHiddenNodes =8;

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("C:\\Users\\hmang\\Desktop\\linear_data_train.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);
        */
        dataLocalPath = "C:\\Users\\hmang\\deeplearning4j-examples\\dl4j-examples\\src\\main\\java\\org\\deeplearning4j\\examples\\dataexamples\\BCW.csv";
        //dataLocalPath="C:\\Users\\hmang\\Desktop\\BCW.csv";
        read_CSV(dataLocalPath);
        set_init_NetworkParams(0.01, 200, 8, 2, 8);
        System.out.println(train);
        /**Now the phase of building the NN model starts */
        try {
            buildNetwork();
            compile();
        }
        catch (Exception obj){System.out.println(obj.getMessage());}


    }

    //The following method is used to read the CSV file
    //And split it into test and train sets for performing further evaluation
    //Also the data members of the class are set
    private static void read_CSV(String csvpath) throws IOException {
        //ADD LOADING ANIMATION HERE
        try {
            RecordReader rr = new CSVRecordReader(',');

            rr.initialize(new FileSplit(new File(csvpath)));
            labelIndex = 8;
            numClasses = 2;
            batchSize = 32;
            System.out.println(rr);
            //Calling method for splitting the file
            train_test_split(rr);
        } catch (InterruptedException e) {
            System.out.println(e.getMessage());
        } catch (Exception e) {
            e.getLocalizedMessage();
        } finally {
            System.out.println("FILE READ");
        }



    }

    private static void train_test_split(RecordReader rr) {
        try {
            //ADD LOADING HERE TOO
            RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
            iterator.setCollectMetaData(true);
            DataSet allData = iterator.next();
            System.out.println(allData);
            allData.shuffle(123); //How to shuffle the data
            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.60);  //Use 70% of data for training

            train = testAndTrain.getTrain();
            test = testAndTrain.getTest();
        } catch (Exception e) {
            System.out.println(e.getCause());
        } finally {
            System.out.println("SPLITTING DONE");
        }
    }

    private static void set_init_NetworkParams(double lr, int nE, int nI, int nO, int nHN) {
        learningRate = lr;
        nEpochs = nE;
        numInputs = nI;
        numOutputs = nO;
        numHiddenNodes = nHN;
        System.out.println("INTIALIZATION DONE");
    }

    private static void buildNetwork() {
        try {
            //Using the nested class of NNC i.e Builder to start building the neural network
                conf = new NeuralNetConfiguration.Builder()
                .seed(0)
                .weightInit(WeightInit.UNIFORM)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)                   //THE FIRST INPUT LAYER
                    .activation(Activation.RELU)
                    .build())
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)                   //THE SECOND LAYER
                    .activation(Activation.LEAKYRELU)
                    .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)      //OUTPUT LAYER
                    .activation(Activation.SOFTMAX)
                    .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        } finally {
            System.out.println("BUILD COMPLETE.....now RETURNING");
        }
    }

    private static void compile() throws  IOException
    {
             model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(100));

                for (int i = 0; i < 1000; i++) {
                    model.fit(train);
                }

            //evaluate the model on the test set
            Evaluation eval = new Evaluation(numClasses);
            INDArray output = model.output(test.getFeatures());
            eval.eval(test.getLabels(), output);

               / save(model);
        }
        public static void save(MultiLayerNetwork model) throws IOException
        {
            try {
                    locationToSave = new File("C:\\Users\\hmang\\deeplearning4j-examples\\dl4j-examples\\src\\main\\java\\org\\deeplearning4j\\examples\\dataexamples\\BCW.zip");
                    //Where to save the network. Note: the file is in .zip format - can be opened externally
                boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
                model.save(locationToSave, saveUpdater);
            }
            catch(Exception obj){System.out.println(obj.getMessage());}
            finally {
                System.out.println("MODEL SAVED.....");
            }
              //MultiLayerNetwork restored = MultiLayerNetwork.load(locationToSave, saveUpdater);

            //System.out.println("Saved and loaded parameters are equal:      " + model.params().equals(restored.params()));
             //locationToSave System.out.println("Saved and loaded configurations are equal:  " + model.getLayerWiseConfigurations().equals(restored.getLayerWiseConfigurations()));
        }
    }

