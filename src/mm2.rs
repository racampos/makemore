use candle_core::{shape, DType, Device, Tensor, D};
use candle_nn::encoding::one_hot;
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap, SGD};
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

    let _words = &mut words
        .iter()
        .map(|word| format!("{}{}", '.', word.to_string()))
        .collect::<Vec<String>>();

    let mut unique_chars = HashSet::new();
    for word in _words.clone() {
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

    let mut xs: Vec<i64> = Vec::new();
    let mut ys: Vec<i64> = Vec::new();
    let block_size = 3;

    for word in words.iter_mut() {
        // println!("{}", word);
        let mut context: Vec<i64> = vec![0; block_size];
        word.push_str(".");
        for ch in word.chars() {
            let ix = stoi.get(&ch).unwrap();
            xs.extend(&context);
            ys.push(*ix);
            let ngram = context.iter().map(|ix| itos.get(ix).unwrap()).join("");
            // println!("{} ---> {}", ngram, itos.get(ix).unwrap());
            context = context[1..].to_vec();
            context.extend(&[*ix]);
        }
    }

    let x = Tensor::from_vec(
        xs.clone(),
        vec![xs.len() / block_size, block_size],
        &Device::Cpu,
    )?;
    let y = Tensor::from_vec(ys.clone(), ys.len(), &Device::Cpu)?;

    let c = Tensor::randn(0f32, 1., (27, 2), &Device::Cpu)?;

    let x0 = x.get_on_dim(1, 0)?;
    let x1 = x.get_on_dim(1, 1)?;
    let x2 = x.get_on_dim(1, 2)?;
    // println!("{:?}", x0.to_vec1::<i64>());

    let emb0 = c.embedding(&x0)?;
    let emb1 = c.embedding(&x1)?;
    let emb2 = c.embedding(&x2)?;

    // println!("{:?}", emb0.to_vec2::<f32>());

    let emb = Tensor::cat(&[emb0, emb1, emb2], 1)?;

    // println!("{:?}", emb.to_vec2::<f32>());
    println!("{:?}", emb.shape());

    struct MultiLevelPerceptron {
        hidden: Linear,
        output: Linear,
    }

    impl MultiLevelPerceptron {
        fn new(vs: VarBuilder) -> candle_core::Result<Self> {
            let hidden = candle_nn::linear(6, 100, vs.pp("hidden"))?;
            let output = candle_nn::linear(100, 27, vs.pp("output"))?;
            Ok(Self { hidden, output })
        }

        fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
            let xs = self.hidden.forward(xs)?;
            let xs = xs.tanh()?;
            self.output.forward(&xs)
        }
    }

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

    let model = MultiLevelPerceptron::new(vs.clone())?;
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), 0.1)?;

    const BSIZE: usize = 32;
    const EPOCHS: i32 = 100;
    let mut avg_loss = 0.0;

    let n_batches = emb.dim(0)? / BSIZE;
    let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();

    for epoch in 0..EPOCHS {
        let mut sum_loss = 0f32;
        batch_idxs.shuffle(&mut thread_rng());
        for batch_idx in batch_idxs.iter() {
            let emb = emb.narrow(0, batch_idx * BSIZE, BSIZE)?;
            let y = y.narrow(0, batch_idx * BSIZE, BSIZE)?;
            let logits = model.forward(&emb)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_sm, &y)?;
            sgd.backward_step(&loss)?;
            sum_loss += loss.to_scalar::<f32>()?;
        }
        avg_loss = sum_loss / n_batches as f32;
        if epoch % 1 == 0 {
            println!("Epoch: {epoch:3} Train loss: {:8.5}", avg_loss);
        }
    }

    println!("\n==========\n");
    println!(
        "Sample results after {} epochs of training with a training loss of {}:\n",
        EPOCHS, avg_loss
    );
    let mut rng = thread_rng();

    for i in 0..10 {
        let mut out: Vec<&char> = Vec::new();
        let mut ctx0 = 0i64;
        let mut ctx1 = 0i64;
        let mut ctx2 = 0i64;
        loop {
            let emb0 = c.embedding(&Tensor::from_vec(vec![ctx0], 1, &Device::Cpu)?)?;
            let emb1 = c.embedding(&Tensor::from_vec(vec![ctx1], 1, &Device::Cpu)?)?;
            let emb2 = c.embedding(&Tensor::from_vec(vec![ctx2], 1, &Device::Cpu)?)?;
            let emb = Tensor::cat(&[emb0, emb1, emb2], 1)?;
            let logits = model.forward(&emb)?;
            let counts = logits.exp()?;
            let counts_sum = counts.sum(1)?.unsqueeze(1)?.expand(&[1, 27])?;
            let probs = counts.div(&counts_sum)?;
            let probs_vec = probs.reshape(&[27])?.to_vec1::<f32>()?;
            let dist = WeightedIndex::new(probs_vec).unwrap();
            ctx0 = ctx1;
            ctx1 = ctx2;
            ctx2 = dist.sample(&mut rng) as i64;
            out.push(itos.get(&ctx2).unwrap());
            if ctx2 == 0 {
                break;
            }
        }
        println!("{:?}", out.iter().join(""));
    }

    Ok(())
}
