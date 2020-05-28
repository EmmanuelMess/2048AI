package ar.com.emmanuelmessulam;

public final class GameEnvironment {

    public final long points;
    public final boolean won;
    public final int[] boardState;

    public GameEnvironment(long points, boolean won, int[] boardState) {
        this.points = points;
        this.won = won;
        this.boardState = boardState;
    }
}
