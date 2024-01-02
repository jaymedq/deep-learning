% Specify the number of samples
N = 360;

% Generate the Bluetooth dataset
bluetoothDataset = generateUraBluetoothDataset(N);

% Display some information about the generated dataset
disp('Bluetooth Dataset Information:');
disp(['Number of Samples: ', num2str(N)]);
details(bluetoothDataset);
disp(['True Angles of Arrival: ', num2str(bluetoothDataset.Angles)]);

% Perform further analysis or visualization as needed
% For example, you can plot the signals and their estimated directions
figure;

subplot(2, 1, 1);
plot(1:N, abs(bluetoothDataset.Signals));
title('Received Signals');
xlabel('Sample');
ylabel('Amplitude');

subplot(2, 1, 2);
stem(bluetoothDataset.Angles, 'r', 'DisplayName', 'True Angles');
hold on;
title('True and Estimated Angles of Arrival');
xlabel('Sample');
ylabel('Angle (degrees)');
legend('True Angles');

