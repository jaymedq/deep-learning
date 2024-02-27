function dataset = generateUlaBluetoothDataset(N)
    % Parameters for the array and signals
    fc = 2.4e9; % Center frequency for Bluetooth in Hz
    c = 3e8; % Speed of light in m/s
    lambda = c/fc; % Wavelength
    numAntennas = 4; % Number of antennas in the array
    arraySpacing = lambda/2; % Antenna spacing in meters

    % Create a uniform linear array
    array = phased.ULA('NumElements', numAntennas, 'ElementSpacing', arraySpacing);

    % Generate angles from 1 to 640 with wraparound from 180 to -180
    step = 5;
    b = 0:step:90;
    angles = b(mod(0:N-1, numel(b)) + 1);

    % Initialize arrays to store results
    musicAngles = zeros(N, 1);

    % Initialize x array
    x = zeros(N, numAntennas);

    % Iterate through each signal and estimate AoA
    for i = 1:N
        % Simulate the received signals for the current angle
        x(i, :) = sensorsig(getElementPosition(array)/lambda, 1, angles(i));%, db2pow(-10));

        % Estimate AoA for the current signal
        musicAoA = phased.MUSICEstimator('SensorArray',array,...
            'OperatingFrequency',fc,'ForwardBackwardAveraging',true,...
            'NumSignalsSource','Property','NumSignals',1,...
            'DOAOutputPort',true);
        [~,musicAngle] = musicAoA(x(i,1:4));

        % Store the estimated AoA for the current signal
        musicAngles(i) = musicAngle;
    end

    % Create the dataset structure
    dataset = struct('Signals', x, 'Angles', angles, 'MusicAngles', musicAngles);
end
