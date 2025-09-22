import numpy as np
import fixed_env as env
import load_trace_norway # change to load_trace_norway or load_trace


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
RESEVOIR = 5  # BB
CUSHION = 10  # BB

CHUNK_TIL_VIDEO_END_CAP = 48.0 
BUFFER_NORM_FACTOR = 10.0

SUMMARY_DIR = './results'
LOG_FILE = './results/log_sim_bb'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward

from collections import deque

all_columns = [
    'a_Last1_chunk_bitrate', 'a_Last1_buffer_size', 'a_Last8_throughput', 'a_Last7_throughput',
    'a_Last6_throughput', 'a_Last5_throughput', 'a_Last4_throughput', 'a_Last3_throughput',
    'a_Last2_throughput', 'a_Last1_throughput', 'a_Last8_downloadtime', 'a_Last7_downloadtime',
    'a_Last6_downloadtime', 'a_Last5_downloadtime', 'a_Last4_downloadtime', 'a_Last3_downloadtime',
    'a_Last2_downloadtime', 'a_Last1_downloadtime', 'a_chunksize1', 'a_chunksize2', 'a_chunksize3',
    'a_chunksize4', 'a_chunksize5', 'a_chunksize6', 'a_Chunks_left', 'a_br',

    

    # 'b_Last1_chunk_bitrate', 'b_Last1_buffer_size', 'b_Last8_throughput', 'b_Last7_throughput',
    # 'b_Last6_throughput', 'b_Last5_throughput', 'b_Last4_throughput', 'b_Last3_throughput',
    # 'b_Last2_throughput', 'b_Last1_throughput', 'b_Last8_downloadtime', 'b_Last7_downloadtime',
    # 'b_Last6_downloadtime', 'b_Last5_downloadtime', 'b_Last4_downloadtime', 'b_Last3_downloadtime',
    # 'b_Last2_downloadtime', 'b_Last1_downloadtime', 'b_chunksize1', 'b_chunksize2', 'b_chunksize3',
    # 'b_chunksize4', 'b_chunksize5', 'b_chunksize6', 'b_Chunks_left', 'b_br', 
    # 'QoE_2'
]


def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace_norway.load_trace()

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    #log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_path = LOG_FILE + '_' + "combined"
    log_file = open(log_path, 'w')
    log_file.write(','.join(all_columns) + '\n')

    epoch = 0
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    r_batch = []

    video_count = 0

    past_throughputs = deque(maxlen=8)
    past_download_times = deque(maxlen=8)
    past_chunk_sizes = deque(maxlen=6)

    last_buffer_size = 0

    chunk_count = 0

    

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        r_batch.append(reward)

        qoe = VIDEO_BIT_RATE[bit_rate] \
                 - REBUF_PENALTY * rebuf \
                 - np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate])

        

        # log time_stamp, bit_rate, buffer_size, reward
        # log_file.write(str(time_stamp / M_IN_K) + '\t' + # seconds
        #                str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
        #                str(buffer_size) + '\t' +
        #                str(rebuf) + '\t' + #seconds
        #                str(video_chunk_size) + '\t' +
        #                str(delay) + '\t' + # milliseconds
        #                str(reward) + '\t' +
        #                str(qoe) + '\n')

        chunk_count += 1
        chunks_left = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        
        log_file.write(
                        str(VIDEO_BIT_RATE[last_bit_rate] / 4300.0) + ',' + # Kbps
                        str(last_buffer_size / BUFFER_NORM_FACTOR) + ',' + # 10 seconds
                        ','.join(['{:.3f}'.format(t) for t in past_throughputs]) + ',' + # Mbps
                        ','.join(['{:.3f}'.format(d / M_IN_K) for d in past_download_times]) + ',' + # seconds
                        ','.join(['{:.2f}'.format(s / M_IN_K / 1024.0) for s in past_chunk_sizes]) + ',' + # in MBytes
                        str(chunks_left) + ',' + # fraction over the total number of chunks
                        str(VIDEO_BIT_RATE[bit_rate]) + '\n' # Kbps
                        )
        log_file.flush()
        # if chunk_count > 3:
        #     log_file.write(
        #                 str(VIDEO_BIT_RATE[last_bit_rate] / 4300.0) + ',' + # Kbps
        #                 str(last_buffer_size / BUFFER_NORM_FACTOR) + ',' + # 10 seconds
        #                 ','.join(['{:.3f}'.format(t) for t in past_throughputs]) + ',' + # Mbps
        #                 ','.join(['{:.3f}'.format(d / M_IN_K) for d in past_download_times]) + ',' + # seconds
        #                 ','.join(['{:.2f}'.format(s / M_IN_K / 1024.0) for s in past_chunk_sizes]) + ',' + # in MBytes
        #                 str(chunks_left) + ',' + # fraction over the total number of chunks
        #                 str(VIDEO_BIT_RATE[bit_rate]) + ',' # Kbps
        #                 )
        #     log_file.flush()
        #     # log_file.write(str(time_stamp / M_IN_K) + '\t' +
        #     if chunk_count > 3 and chunk_count % 2 == 1:
        #         last_q = np.log(VIDEO_BIT_RATE[last_bit_rate] / VIDEO_BIT_RATE[0])
        #         cur_q = np.log(VIDEO_BIT_RATE[bit_rate] / VIDEO_BIT_RATE[0])
        #         qoe_2 =  last_q + cur_q - np.abs(last_q - cur_q)
        #         log_file.write(str(qoe_2) + '\n')
        #         log_file.flush()

        last_bit_rate = bit_rate

        last_buffer_size = buffer_size

        # add current throughput to the past throughput
        curr_throughput = (video_chunk_size * 8.0) / delay / 1e6
        past_throughputs.append(curr_throughput)

        # add current chunk size to the past chunk size
        past_chunk_sizes.append(video_chunk_size)

        # add current download time to the past download time
        curr_download_time = delay
        past_download_times.append(curr_download_time)


        if buffer_size < RESEVOIR:
            bit_rate = 0
        elif buffer_size >= RESEVOIR + CUSHION:
            bit_rate = A_DIM - 1
        else:
            bit_rate = (A_DIM - 1) * (buffer_size - RESEVOIR) / float(CUSHION)

        bit_rate = int(bit_rate)

        if end_of_video:
            # log_file.write('\n')
            # log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            r_batch = []

            print("video count", video_count)
            video_count += 1

            if video_count > len(all_file_names):
                break

            # reset chunk count
            chunk_count = 0

            # log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            # log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
