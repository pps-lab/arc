
use ark_std::perf_trace::TimerInfo;
use mpc_net::Stats;

#[macro_export]
macro_rules! end_timer {
    ($time:expr) => {{
        end_timer!($time, || "");
    }};
    ($time:expr, $msg:expr) => {{

        let time = $time.time;
        let final_time = time.elapsed();
        let final_time = {
            // let millis = final_time.subsec_millis();
            let micros = final_time.as_micros();
            let message = format!("{} {}", $time.msg, $msg());

            println!("TIMER (name={}_mus) (value={})", message.trim_end(), micros);

            // // write to TRACE_FILENAME
            // {
            //     let mut file = std::fs::OpenOptions::new()
            //         .append(true)
            //         .create(true)
            //         .open(TRACE_FILENAME)
            //         .unwrap();
            //     let line = format!("\"{}\",{}\n", message.trim_end(), micros);
            //     file.write_all(line.as_bytes()).unwrap();
            //     file.flush()
            // }
        };

    }};
}

pub fn print_stats(stats: Stats) {
    println!("STATS (name=bytes_sent) (value={})", stats.bytes_sent);
    println!("STATS (name=bytes_recv) (value={})", stats.bytes_recv);
    println!("STATS (name=broadcasts) (value={})", stats.broadcasts);
    println!("STATS (name=to_king) (value={})", stats.to_king);
    println!("STATS (name=from_king) (value={})", stats.from_king);
}

pub fn print_global_stats(global_data_sent: usize) {
    println!("STATS (name=global_bytes) (value={})", global_data_sent);
}