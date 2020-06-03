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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
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
            .weightDecay(0.0001)
            .list()
            .layer(new DenseLayer.Builder()
                    .nIn(16).nOut(8)
                    .build())
            .layer(new OutputLayer.Builder()
                    .nIn(8).nOut(4)
                    .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
                    .build())
            .build();
    MultiLayerNetwork Qnetwork = new MultiLayerNetwork(conf);

    private GameEnvironment oldState;
    private GameEnvironment currentState;
    private INDArray oldQuality;

    private GameAction lastAction;

    public SimpleAgent() {
        Qnetwork.init();
        ui();
    }

    public void setCurrentState(GameEnvironment currentState) {
        this.currentState = currentState;
    }

    private final ArrayList<INDArray> input = new ArrayList<>();
    private final ArrayList<INDArray> output = new ArrayList<>();
    private final ArrayList<Double> rewards = new ArrayList<>();

    private int epsilon = 100;

    public GameAction act() {
        if(oldState != null) {
            double reward = currentState.points - oldState.points;

            if (currentState.lost) {
                reward = 0;
            }

            input.add(oldState.boardState);
            output.add(oldQuality);
            rewards.add(reward);

            if (currentState.lost || input.size() == 20) {
                for(int i = 0; i < rewards.size(); i++) {
                    double discount = 1.4;
                    double discountedReward = 0;

                    for(int j = i; j < rewards.size(); j++) {
                        discountedReward += rewards.get(j) * Math.pow(discount, j - i);
                    }

                    rewards.set(i, lerp(discountedReward, 1024));
                }

                ArrayList<DataSet> dataSets = new ArrayList<>();

                for(int i = 0; i < input.size(); i++) {
                    INDArray correctOut = output.get(i).putScalar(lastAction.ordinal(), rewards.get(i));

                    dataSets.add(new DataSet(input.get(i), correctOut));
                }

                Qnetwork.fit(DataSet.merge(dataSets));

                input.clear();
                output.clear();
                rewards.clear();
            }

            epsilon = Math.max(1, epsilon - 10);
        }

        oldState = currentState;
        oldQuality = Qnetwork.output(currentState.boardState);

        GameAction action;


        if(random.nextInt(100) < 100-epsilon) {
            action = GameAction.values()[oldQuality.argMax(1).getInt()];
        } else {
            action = GameAction.values()[new Random().nextInt(GameAction.values().length)];
        }

        lastAction = action;

        return action;
    }

    private static double lerp(double x, int maxVal) {
        return x/maxVal;
    }

    private void ui() {
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        Qnetwork.setListeners(new StatsListener(statsStorage));
    }
}
