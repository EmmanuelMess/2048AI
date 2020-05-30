package ar.com.emmanuelmessulam;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

import static com.bulenkov.game2048.Game2048.SEED;

public class SimpleAgent {
    private static final Random random = new Random(SEED);

    private static final MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(SEED)
            .weightInit(WeightInit.XAVIER)
            .updater(new AdaGrad(0.5))
            .activation(Activation.RELU)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .l2(0.0001)
            .list()
            //First hidden layer
            .layer(0, new DenseLayer.Builder()
                    .nIn(16).nOut(12)
                    .build())
            .layer(1, new DenseLayer.Builder()
                    .nIn(12).nOut(6)
                    .build())
            .layer(2, new DenseLayer.Builder()
                    .nIn(6).nOut(4)
                    .build())
            //Output layer
            .layer(3, new OutputLayer.Builder()
                    .nIn(4).nOut(4)
                    .activation(Activation.SOFTMAX)
                    .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .build())
            .build();
    MultiLayerNetwork Qnetwork = new MultiLayerNetwork(conf);

    private GameEnvironment oldState;
    private GameEnvironment currentState;

    private GameAction lastAction;

    public SimpleAgent() {
        Qnetwork.init();
        ui();
    }

    public void setCurrentState(GameEnvironment currentState) {
        this.currentState = currentState;
    }

    public GameAction act() {
        if(oldState != null) {
            int oldPoints = oldState.points;
            double reward = lerp(currentState.points - oldPoints, 2048);

            if(currentState.lost) {
                reward = 0;

                System.out.println("Lost: " + currentState.points);
            }

            INDArray oldQuality = Qnetwork.output(oldState.boardState);
            INDArray realQuality = oldQuality.add(0).putScalar(lastAction.ordinal(), reward);

            Qnetwork.fit(oldState.boardState, realQuality);//TODO actually use minibatching
        }

        oldState = currentState;

        INDArray result = Qnetwork.output(currentState.boardState);
        GameAction action = GameAction.values()[result.argMax(1).getInt()];

        lastAction = action;

        return action;
    }

    private static double lerp(double x, int maxVal) {
        return x/maxVal;
    }

    private void ui() {
        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Then add the StatsListener to collect this information from the network, as it trains
        Qnetwork.setListeners(new StatsListener(statsStorage));
    }
}
