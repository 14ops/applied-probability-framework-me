import javax.swing.*;
import java.awt.*;

public class GameControlPanel extends JFrame {

    public GameControlPanel() {
        setTitle("Game Control Panel");
        setSize(600, 400);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        // Create components
        JButton startButton = new JButton("Start Simulation");
        JTextArea outputArea = new JTextArea();
        outputArea.setEditable(false);

        // Layout
        setLayout(new BorderLayout());
        add(startButton, BorderLayout.NORTH);
        add(new JScrollPane(outputArea), BorderLayout.CENTER);

        // Action Listener
        startButton.addActionListener(e -> {
            outputArea.setText("Simulation started...\n");
            // Here you would add the logic to connect to the Python backend
        });
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new GameControlPanel().setVisible(true);
        });
    }
}
