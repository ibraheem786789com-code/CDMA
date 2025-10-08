clc; clear; close all;

%% ----------------- Parameters -----------------
Fs = 8000;              
num_bits = 8;           
levels = 2^num_bits;    
step_size = 2 / (levels - 1);  

SF = 8;                
user_id = 3;            
frameLenSec = 0.01;     
samplesPerFrame = round(Fs * frameLenSec);

snr_db = 20;            
add_interference = true;
intf_amp = 0.12;        

display_every_frames = 20;  
plotBufLenSamples = 800;    
plotBufLenBits = 800;       

%% ----------------- Walsh Codes -----------------
if mod(log2(SF),1) ~= 0
    error('SF must be a power of two for hadamard.');
end
W = hadamard(SF);
code = W(user_id, :)';

other_row = mod(user_id, SF) + 1;
if other_row == user_id, other_row = mod(user_id+1, SF)+1; end
code_intf = W(other_row, :)';

%% ----------------- Audio I/O -----------------
deviceReader = audioDeviceReader('SampleRate', Fs, 'SamplesPerFrame', samplesPerFrame);
deviceWriter = audioDeviceWriter('SampleRate', Fs);

disp('--- Real-time CDMA pipeline (8-bit PCM + LMS AEC) ---');
disp('Speak into microphone. Press Ctrl+C to stop.');

%% ----------------- LMS Echo Canceller -----------------
M = 128;                      % Filter length (adjust if needed)
mu = 0.001;                   % Step size for LMS
lms_weights = zeros(M, 1);    % Initialize adaptive filter weights
x_play_hist = zeros(M, 1);    % Playback history buffer

%% ----------------- Helpers -----------------
bitWeights = (2.^(num_bits-1:-1:0))';
frameCount = 0;
ringSize = 200;
berRing = nan(ringSize,1);
snrRing = nan(ringSize,1);
ringIdx = 0;

% Buffers for plotting
txBuf = zeros(0,1);
rxBuf = zeros(0,1);
intfTxBuf = zeros(0,1);
intfRxBuf = zeros(0,1);

%% ----------------- Plot Setup -----------------
fig = figure('Name','CDMA Live + LMS Echo Canceller','NumberTitle','off');
ax1 = subplot(2,2,1); h1 = plot(ax1, nan, nan); title(ax1,'Mic Input'); grid(ax1,'on');
ax2 = subplot(2,2,2); h2 = plot(ax2, nan, nan); title(ax2,'Interference Bits'); grid(ax2,'on');
ax3 = subplot(2,2,3); h3 = plot(ax3, nan, nan); title(ax3,'Recovered (Cleaned)'); grid(ax3,'on');
ax4 = subplot(2,2,4); h4 = plot(ax4, nan, nan); title(ax4,'Recovered Interference'); grid(ax4,'on');

%% ----------------- Main Loop -----------------
try
    while true
        frameCount = frameCount + 1;

        %% 1) Microphone input
        x = double(deviceReader());
        x = x(:);
        if max(abs(x)) < 0.01
            x(:) = 0;
        end

        %% 2) Quantize and convert to bits
        xq = round((x + 1) / step_size);
        xq = min(max(xq, 0), levels - 1);
        bin_stream = de2bi(xq, num_bits, 'left-msb');
        tx_bits = reshape(bin_stream.', [], 1);

        %% 3) BPSK modulation
        tx_symbols = 2*double(tx_bits) - 1;

        %% 4) Spread with Walsh
        tx_chips = kron(tx_symbols, code);

        %% 5) Add interference
        if add_interference
            intf_bits = 2*(randi([0 1], size(tx_symbols))) - 1;
            intf_chips = kron(intf_bits, code_intf) * intf_amp;
        else
            intf_bits = zeros(size(tx_symbols));
            intf_chips = zeros(size(tx_chips));
        end

        %% 6) AWGN channel
        tx_combined = tx_chips + intf_chips;
        P_signal = mean(tx_combined.^2);
        noise_var = P_signal / (10^(snr_db/10));
        noise = sqrt(noise_var) * randn(size(tx_combined));
        rx_chips = tx_combined + noise;

        %% 7) Despread
        num_symbols = length(tx_symbols);
        rx_matrix = reshape(rx_chips, SF, num_symbols);
        decisions = code' * rx_matrix;
        rx_bits = decisions.' > 0;

        %% 8) Reconstruct audio samples
        rx_bits = rx_bits(1:num_symbols);
        rx_bits_mat = reshape(rx_bits, num_bits, []).';
        rx_dec = double(rx_bits_mat) * bitWeights;
        x_rec = rx_dec * step_size - 1;

       
        %% 9) LMS Echo Cancellation (frame-wise adaptive)
cleaned = zeros(size(x_rec));

for n = 1:length(x_rec)
    % Update input buffer (playback history)
    x_play_hist = [x_rec(n); x_play_hist(1:end-1)];
    
    % Estimated echo
    y_hat = lms_weights' * x_play_hist;
    
    % Error = mic - estimated echo
    e = x(n) - y_hat;
    cleaned(n) = e;
    
    % Adapt weights
    lms_weights = lms_weights + mu * x_play_hist * e;
end

% Use cleaned signal for playback
deviceWriter(cleaned);


        

        %% ----- Plotting -----
        txBuf = [txBuf; x];
        if length(txBuf) > plotBufLenSamples, txBuf = txBuf(end-plotBufLenSamples+1:end); end
        rxBuf = [rxBuf; cleaned];

        if length(rxBuf) > plotBufLenSamples, rxBuf = rxBuf(end-plotBufLenSamples+1:end); end
        intfTxBuf = [intfTxBuf; intf_bits];
        if length(intfTxBuf) > plotBufLenBits, intfTxBuf = intfTxBuf(end-plotBufLenBits+1:end); end

        decisions_intf = code_intf' * rx_matrix;
        rx_bits_intf = decisions_intf.' > 0;
        intfRxBuf = [intfRxBuf; rx_bits_intf];
        if length(intfRxBuf) > plotBufLenBits, intfRxBuf = intfRxBuf(end-plotBufLenBits+1:end); end

        if mod(frameCount, display_every_frames) == 0
            set(h1, 'XData', 1:length(txBuf), 'YData', txBuf);
            set(h2, 'XData', 1:length(intfTxBuf), 'YData', intfTxBuf);
            set(h3, 'XData', 1:length(rxBuf), 'YData', rxBuf);
            set(h4, 'XData', 1:length(intfRxBuf), 'YData', double(intfRxBuf));
            drawnow limitrate;
        end

        %% 11) BER & SNR
        ncomp = min(length(tx_bits), length(rx_bits));
        BER_frame = sum(tx_bits(1:ncomp) ~= double(rx_bits(1:ncomp))) / ncomp;

        sig_len = min(length(x), length(cleaned));
signal_power = mean(x(1:sig_len).^2);
noise_power = mean((x(1:sig_len) - cleaned(1:sig_len)).^2);

        SNR_meas = 10*log10(signal_power / (noise_power + eps));

        ringIdx = mod(ringIdx, ringSize) + 1;
        berRing(ringIdx) = BER_frame;
        snrRing(ringIdx) = SNR_meas;

        if mod(frameCount, display_every_frames) == 0
            fprintf('Frame %5d | Avg BER = %.3e | Avg SNR = %.2f dB\n', ...
                frameCount, nanmean(berRing), nanmean(snrRing));
        end
    end
catch ME
    try release(deviceReader); catch, end
    try release(deviceWriter); catch, end
    if ~strcmp(ME.identifier, 'MATLAB:class:InvalidHandle')
        rethrow(ME);
    end
end

release(deviceReader);
release(deviceWriter);
disp('Stopped.');