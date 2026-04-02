package com.example.executorchedgeaiinference

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

import org.pytorch.executorch.Tensor
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module

import java.io.File
import java.io.FileOutputStream
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer

class MainActivity : AppCompatActivity() {

    private lateinit var module: Module
    private lateinit var resultText: TextView

    lateinit var tokenizer: ai.djl.huggingface.tokenizers.HuggingFaceTokenizer

    override fun onCreate(savedInstanceState: Bundle?) {
        tokenizer = ai.djl.huggingface.tokenizers.HuggingFaceTokenizer.newInstance(
            assetFilePath("tokenizer.json")
        )
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        resultText = findViewById(R.id.resultText)

        module = Module.load(assetFilePath("smollm_135m_fixed.pte"))
        //module = Module.load(assetFilePath("simple_model.pte"))

        val button = findViewById<Button>(R.id.runButton)
        button.setOnClickListener {
            runInference("Hello")
        }
    }

    /*private fun runInference() {
        val inputTensor = Tensor.fromBlob(
            FloatArray(1 * 3 * 224 * 224) { 0.5f },
            longArrayOf(1, 3, 224, 224)
        )

        // ✅ Use EValue instead of IValue
        val input = EValue.from(inputTensor)

        val output = module.forward(input)[0].toTensor()
        val scores = output.dataAsFloatArray

        val maxIdx = scores.indices.maxByOrNull { scores[it] } ?: -1
        resultText.text = "Predicted class index: $maxIdx"
    }
    */
    private fun runInference(text: String): String {
        try {
            val encoding = tokenizer.encode(text)
            val ids = encoding.ids

            val SEQ_LEN = 32

            val inputIds = LongArray(SEQ_LEN) { 0L }
            val attentionMask = LongArray(SEQ_LEN) { 0L }

            val len = minOf(ids.size, SEQ_LEN)

            for (i in 0 until len) {
                inputIds[i] = ids[i].toLong()
                attentionMask[i] = 1L
            }

            // ✅ MUST be (1, 32)
            val inputTensor = Tensor.fromBlob(
                inputIds,
                longArrayOf(1, SEQ_LEN.toLong())
            )

            val maskTensor = Tensor.fromBlob(
                attentionMask,
                longArrayOf(1, SEQ_LEN.toLong())
            )

            // 🔥 CRITICAL: pass BOTH tensors
            Log.d("DEBUG", "inputIds size = ${inputIds.size}")
            Log.d("DEBUG", "mask size = ${attentionMask.size}")
            val output = module.forward(EValue.from(inputTensor), EValue.from(maskTensor))
            //val output = module.forward(inputTensor, maskTensor)

            val outputTensor = output[0].toTensor()
            val logits = outputTensor.dataAsFloatArray

            return "Success: ${logits.size}"

        } catch (e: Exception) {
            e.printStackTrace()
            return "Error: ${e.message}"
        }
    }

    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
            }
        }
        return file.absolutePath
    }
}
