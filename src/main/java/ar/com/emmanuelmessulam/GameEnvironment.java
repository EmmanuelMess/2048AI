package ar.com.emmanuelmessulam;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

public final class GameEnvironment {

    public final int points;
    public final boolean lost;
    public final INDArray boardState;

    public GameEnvironment(int points, boolean lost, int[] boardState) {
        this.points = points;
        this.lost = lost;
        this.boardState = new NDArray(boardState, new int[] {1, 16}, new int[] {16, 1});
    }
}
