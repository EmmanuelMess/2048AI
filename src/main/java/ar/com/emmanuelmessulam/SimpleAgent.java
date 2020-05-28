package ar.com.emmanuelmessulam;

import java.util.Random;

public class SimpleAgent {

    private GameEnvironment environment;

    public void setEnvironment(GameEnvironment environment) {
        this.environment = environment;
    }

    public GameAction act() {


        return GameAction.values()[new Random().nextInt(GameAction.values().length)];
    }
}
