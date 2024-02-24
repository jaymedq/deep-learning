function dataset = generateBluetoothDataset(N)
    % Parameters for the array and signals
    fc = 2.4e9; % Center frequency for Bluetooth in Hz
    c = 3e8; % Speed of light in m/s
    lambda = c/fc; % Wavelength
    numAntennas = 8; % Number of antennas in the array
    arraySpacing = lambda/2; % Antenna spacing in meters

    % Create a uniform linear array
    array = phased.ULA('NumElements', numAntennas, 'ElementSpacing', arraySpacing);

    % Generate random angles for incoming signals
    angles = randi([-180, 180], 1, N);

    % Simulate the received signals
    x = sensorsig(getElementPosition(array)/lambda, N, angles, db2pow(-10));

    % Create the dataset structure
    dataset = struct('Signals', x, 'Angles', angles);
end
