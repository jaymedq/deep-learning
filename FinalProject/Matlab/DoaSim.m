% Specify the number of samples (Take X as number of turns on the axis 360 * X + X)
N = 90;%18100;

% Generate the Bluetooth dataset
bluetoothDataset = generateUraBluetoothDataset(N);
%bluetoothDataset = generateUlaBluetoothDataset(N);

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
writematrix(bluetoothDataset.Signals', ['Signals',csvFileName]);
writematrix(bluetoothDataset.Angles', ['Angles',csvFileName]);
writematrix(bluetoothDataset.MusicAngles', ['MusicAngles',csvFileName]);

subplot(2, 1, 1);
plot(1:N, abs(bluetoothDataset.Signals));
title('Received Signals');
xlabel('Sample N');
ylabel('Complex Magnitude');

subplot(2, 1, 2);
hold on;

stem(bluetoothDataset.Angles);
stem(bluetoothDataset.MusicAngles);

hold off;
title('True and Estimated Direction of Arrival');
%title('Direction of Arrival');
xlabel('Sample N');
ylabel('Angle (degrees)');
legend('True Angles', 'Music Angles');


