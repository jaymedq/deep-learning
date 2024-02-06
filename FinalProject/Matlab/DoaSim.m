% Specify the number of samples (Take X as number of turns on the axis 360 * X + X)
N = 27075;

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

% Generate a string with the current datetime
datetimeString = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

% Save the dataset as CSV with datetime in the title
csvFileName = ['bluetooth_signals_dataset_', datetimeString, '.csv'];
writematrix([bluetoothDataset.Angles', bluetoothDataset.Signals], csvFileName);
% writematrix(, 'bluetooth_angles_dataset.csv');

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

