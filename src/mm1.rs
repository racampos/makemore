use candle_core::{DType, Device, Tensor, D};
use candle_nn::encoding::one_hot;
use candle_nn::{loss, ops, Module, Optimizer, VarBuilder, VarMap, SGD};
use itertools::Itertools;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn train() -> Result<(), Box<dyn std::error::Error>> {
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

    let mut unique_chars = HashSet::new();
    for word in words {
        for ch in word.chars() {
            unique_chars.insert(ch);
        }
    }
    let mut chars: Vec<char> = unique_chars.into_iter().collect();
    chars.sort();

    let stoi =
        HashMap::<char, i64>::from_iter(chars.iter().enumerate().map(|(i, &c)| (c, i as i64)));

    let itos =
        HashMap::<i64, char>::from_iter(chars.iter().enumerate().map(|(i, &c)| (i as i64, c)));

    let mut b: HashMap<(char, char), i32> = HashMap::new();

    let mut xs: Vec<i64> = Vec::new();
    let mut ys: Vec<i64> = Vec::new();

    for word in words.iter() {
        for (ch1, ch2) in word.chars().zip(word.chars().skip(1)) {
            let ix1 = stoi.get(&ch1).unwrap();
            let ix2 = stoi.get(&ch2).unwrap();
            let bigram = (ch1, ch2);
            b.insert(bigram, b.get(&bigram).unwrap_or(&0) + 1);
            xs.push(*ix1);
            ys.push(*ix2);
        }
    }

    let xs_len = xs.len();
    let ys_len = ys.len();
    let xs = Tensor::from_vec(xs.clone(), xs_len, &Device::Cpu)?;
    let ys = Tensor::from_vec(ys.clone(), ys_len, &Device::Cpu)?;

    let xenc = one_hot(xs.clone(), 27, 1f32, 0f32);
    let xenc = xenc.unwrap();
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let model = candle_nn::linear(27, 27, vs)?;
    let mut sgd = SGD::new(varmap.all_vars(), 1.0)?;
    const EPOCHS: i32 = 250;
    let mut loss_scalar = 0.0;

    for epoch in 0..EPOCHS {
        let logits = model.forward(&xenc)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &ys)?;
        sgd.backward_step(&loss)?;
        loss_scalar = loss.to_scalar::<f32>()?;
        if epoch % 10 == 0 {
            println!("Epoch: {epoch:3} Train loss: {:8.5}", loss_scalar);
        }
    }

    println!("\n==========\n");
    println!(
        "Sample results after {} epochs of training with a training loss of {}:\n",
        EPOCHS, loss_scalar
    );
    let mut rng = thread_rng();

    for i in 0..10 {
        let mut out: Vec<&char> = Vec::new();
        let mut ix: i64 = 0;
        loop {
            let ix_tensor = Tensor::from_vec(vec![ix], 1, &Device::Cpu)?;
            let xenc = one_hot(ix_tensor, 27, 1f32, 0f32)?;
            let logits = model.forward(&xenc)?;
            let counts = logits.exp()?;
            let counts_sum = counts.sum(1)?.unsqueeze(1)?.expand(&[1, 27])?;
            let probs = counts.div(&counts_sum)?;
            let probs_vec = probs.reshape(&[27])?.to_vec1::<f32>()?;
            let dist = WeightedIndex::new(probs_vec).unwrap();
            ix = dist.sample(&mut rng) as i64;
            out.push(itos.get(&ix).unwrap());
            if ix == 0 {
                break;
            }
        }
        println!("{:?}", out.iter().join(""));
    }

    Ok(())
}
