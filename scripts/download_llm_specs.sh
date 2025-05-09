export GIT_LFS_SKIP_SMUDGE=1

mkdir -p data/llmspecs
cd data/llmspecs
git clone https://hf-mirror.com/TheBloke/Llama-2-7B-Chat-GPTQ
git clone https://hf-mirror.com/TheBloke/Llama-2-13B-Chat-GPTQ
git clone https://hf-mirror.com/TheBloke/Llama-2-70B-Chat-GPTQ
git clone https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
git clone https://hf-mirror.com/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4
git clone https://hf-mirror.com/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4
git clone https://hf-mirror.com/Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4

# Unified test bed
cd Llama-2-70B-Chat-GPTQ
git checkout gptq-4bit-128g-actorder_True
sed -i 's/"desc_act": true/"desc_act": false/' "config.json"
cd ..

cd ../..
cp -r data/llmspecs data/llmspecs_2bit
cd data/llmspecs_2bit

for dir in */; do
    if [ -f "$dir/config.json" ]; then
        sed -i 's/"bits": 4/"bits": 2/' "$dir/config.json"
        echo "Updated bits in $dir/config.json"
    else
        echo "No config.json found in $dir"
    fi
done

cd ../..
