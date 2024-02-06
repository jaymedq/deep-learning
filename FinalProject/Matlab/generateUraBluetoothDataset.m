function dataset = generateUraBluetoothDataset(N)
    % Parameters for the array and signals
    fc = 2.4e9; % Center frequency for Bluetooth in Hz
    c = 3e8; % Speed of light in m/s
    lambda = c/fc; % Wavelength
    arraySize = [4 4]; % Number of antennas in the array
    arraySpacing = lambda/2; % Antenna spacing in meters

    % Create a uniform linear array
    array = phased.URA('Size', arraySize, 'ElementSpacing', arraySpacing);

    % Generate angles from 1 to 640 with wraparound from 180 to -180
    angles = mod(0:N-1, 361) - 180;

    % Simulate the received signals
    x = sensorsig(getElementPosition(array)/lambda, N, angles, db2pow(-10));

    % Create the dataset structure
    dataset = struct('Signals', x, 'Angles', angles);
end
