use candle_core::{DType, Device, IndexOp, Tensor};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::Hash;
use std::io::{self, BufRead, BufReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "names.txt";
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut words = Vec::new();
    for line in reader.lines() {
        let line = line?;
        words.push(line);
    }

    let words = &words
        .iter()
        .map(|word| format!("{}{}{}", '.', word.to_string(), '.'))
        .collect::<Vec<String>>();

    let N = Tensor::zeros((27, 27), DType::I64, &Device::Cpu)?;

    let mut unique_chars = HashSet::new();
    for word in words {
        for ch in word.chars() {
            unique_chars.insert(ch);
        }
    }
    let mut chars: Vec<char> = unique_chars.into_iter().collect();
    chars.sort();

    let stoi =
        HashMap::<char, i32>::from_iter(chars.iter().enumerate().map(|(i, &c)| (c, i as i32)));

    let mut b: HashMap<(char, char), i32> = HashMap::new();

    for word in words.iter() {
        for (ch1, ch2) in word.chars().zip(word.chars().skip(1)) {
            let ix1 = stoi.get(&ch1).unwrap();
            let ix2 = stoi.get(&ch2).unwrap();
            N.slice_assign(ranges, src)
            let bigram = (ch1, ch2);
            b.insert(bigram, b.get(&bigram).unwrap_or(&0) + 1);
        }
    }

    Ok(())
}
