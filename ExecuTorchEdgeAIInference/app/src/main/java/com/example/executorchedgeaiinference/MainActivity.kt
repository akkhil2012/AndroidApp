package com.example.phi3chat

import android.os.Bundle
import android.util.Log
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.EditText
import android.widget.ImageButton
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.facebook.soloader.SoLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pytorch.executorch.LlamaCallback
import org.pytorch.executorch.LlamaModule
// Updated imports for ExecuTorch SDK 0.5.0+
//import org.pytorch.executorch.sdk.LlamaCallback
//import org.pytorch.executorch.sdk.LlamaModule
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {

    // ── Constants ─────────────────────────────────────────────────────────────
    companion object {
        private const val TAG            = "EdgeApp_Phi3"
        private const val MODEL_FILE     = "phi3_mini_8da4w.pte"
        private const val TOKENIZER_FILE = "tokenizer.json"
        private const val MAX_NEW_TOKENS = 256
        private const val TEMPERATURE    = 0.7f
        private const val SYSTEM_PROMPT  =
            "You are a helpful AI assistant running locally on an Android device. " +
            "Keep your responses concise and accurate."
    }

    // ── UI ────────────────────────────────────────────────────────────────────
    private lateinit var recyclerView   : RecyclerView
    private lateinit var messageInput   : EditText
    private lateinit var sendButton     : ImageButton
    private lateinit var clearButton    : ImageButton
    private lateinit var statusText     : TextView
    private lateinit var chatAdapter    : ChatAdapter

    // ── Model ─────────────────────────────────────────────────────────────────
    private var llamaModule    : LlamaModule? = null
    private var isModelLoaded  = false
    private var isGenerating   = false

    // ── Conversation history for multi-turn chat ──────────────────────────────
    private val conversationHistory = StringBuilder()

    // ── Lifecycle ─────────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // MUST be first — initialises ExecuTorch native libraries
        SoLoader.init(this, false)

        setContentView(R.layout.activity_main)
        setupUI()
        loadModelAsync()
    }

    override fun onDestroy() {
        super.onDestroy()
        llamaModule?.stop()
        llamaModule = null
    }

    // ── UI Setup ──────────────────────────────────────────────────────────────
    private fun setupUI() {
        recyclerView = findViewById(R.id.chat_recycler)
        messageInput = findViewById(R.id.message_input)
        sendButton   = findViewById(R.id.send_button)
        clearButton  = findViewById(R.id.clear_button)
        statusText   = findViewById(R.id.status_text)

        chatAdapter = ChatAdapter()
        recyclerView.apply {
            layoutManager = LinearLayoutManager(this@MainActivity).apply {
                stackFromEnd = true          // new messages appear at bottom
            }
            adapter = chatAdapter
        }

        // Disable input until model is ready
        setInputEnabled(false)

        sendButton.setOnClickListener { sendMessage() }

        clearButton.setOnClickListener {
            conversationHistory.clear()
            chatAdapter.clearMessages()
            chatAdapter.addMessage(
                ChatMessage("Chat cleared. Ask me anything!", isUser = false)
            )
        }

        messageInput.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEND) {
                sendMessage(); true
            } else false
        }
    }

    private fun setInputEnabled(enabled: Boolean) {
        messageInput.isEnabled = enabled
        sendButton.isEnabled   = enabled
        messageInput.hint      = if (enabled) "Ask me anything..." else "Loading model..."
    }

    // ── Model Loading ─────────────────────────────────────────────────────────
    private fun loadModelAsync() {
        statusText.text = "⏳ Loading Phi-3 Mini…"

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Check available memory before loading
                val freeMB = checkAvailableMemoryMB()
                Log.d(TAG, "Available memory before load: ${freeMB}MB")
                if (freeMB < 512) {
                    throw OutOfMemoryError("Insufficient memory: ${freeMB}MB free. Need 512MB+")
                }

                // Copy assets from APK to internal storage
                val modelPath     = copyAssetToFilesDir(MODEL_FILE)
                val tokenizerPath = copyAssetToFilesDir(TOKENIZER_FILE)

                Log.d(TAG, "Model path     : $modelPath")
                Log.d(TAG, "Tokenizer path : $tokenizerPath")

                // Initialise ExecuTorch LlamaModule
                llamaModule = LlamaModule(
                    modelPath,
                    tokenizerPath,
                    TEMPERATURE,
                )

                // Warm-up pass — primes caches, reduces first-query latency
                Log.d(TAG, "Running warm-up pass…")
                llamaModule?.generate(
                    "Hi",
                    8,
                    object : LlamaCallback {
                        override fun onResult(token: String?) { /* discard warm-up tokens */ }
                        override fun onStats(tps: Float) {}
                    }
                )

                withContext(Dispatchers.Main) {
                    isModelLoaded   = true
                    statusText.text = "✅ Phi-3 Mini ready"
                    setInputEnabled(true)

                    chatAdapter.addMessage(
                        ChatMessage(
                            "Hi! I'm Phi-3 Mini running entirely on your device — " +
                            "no internet needed. Ask me anything!",
                            isUser = false
                        )
                    )
                    Log.d(TAG, "✅ Model loaded and warmed up")
                }

            } catch (e: Exception) {
                Log.e(TAG, "❌ Model load failed: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    statusText.text = "❌ Load failed"
                    chatAdapter.addMessage(
                        ChatMessage(
                            "⚠️ Failed to load model:\n${e.message}\n\n" +
                            "Make sure phi3_mini.pte and tokenizer.json " +
                            "are in the app/src/main/assets/ folder.",
                            isUser = false
                        )
                    )
                }
            }
        }
    }

    // ── Message Sending ───────────────────────────────────────────────────────
    private fun sendMessage() {
        val userText = messageInput.text.toString().trim()
        if (userText.isEmpty() || !isModelLoaded || isGenerating) return

        // Clear input and hide keyboard
        messageInput.text.clear()
        hideKeyboard()

        // Show user bubble
        chatAdapter.addMessage(ChatMessage(userText, isUser = true))
        scrollToBottom()

        // Placeholder bot bubble (streaming)
        chatAdapter.addMessage(ChatMessage("", isUser = false, isStreaming = true))

        val prompt          = buildPhi3Prompt(userText)
        val generatedTokens = StringBuilder()
        isGenerating        = true
        sendButton.isEnabled = false
        statusText.text     = "⚡ Generating…"

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                llamaModule?.generate(
                    prompt,
                    MAX_NEW_TOKENS,
                    object : LlamaCallback {

                        override fun onResult(token: String?) {
                            token?.let {
                                generatedTokens.append(it)
                                lifecycleScope.launch(Dispatchers.Main) {
                                    chatAdapter.updateLastMessage(
                                        generatedTokens.toString(),
                                        isStreaming = true
                                    )
                                    scrollToBottom()
                                }
                            }
                        }

                        override fun onStats(tokensPerSecond: Float) {
                            lifecycleScope.launch(Dispatchers.Main) {
                                statusText.text =
                                    "⚡ %.1f tok/s".format(tokensPerSecond)
                            }
                        }
                    }
                )

                // Generation complete — finalise bubble
                withContext(Dispatchers.Main) {
                    val finalText = generatedTokens.toString()
                    chatAdapter.updateLastMessage(finalText, isStreaming = false)

                    // Append to history for multi-turn context
                    conversationHistory.append(
                        "<|user|>\n$userText<|end|>\n" +
                        "<|assistant|>\n$finalText<|end|>\n"
                    )

                    isGenerating         = false
                    sendButton.isEnabled = true
                    statusText.text      = "✅ Phi-3 Mini ready"
                    scrollToBottom()
                }

            } catch (e: Exception) {
                Log.e(TAG, "❌ Generation failed: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    chatAdapter.updateLastMessage(
                        "⚠️ Generation error: ${e.message}",
                        isStreaming = false
                    )
                    isGenerating         = false
                    sendButton.isEnabled = true
                    statusText.text      = "❌ Generation failed"
                }
            }
        }
    }

    // ── Phi-3 Prompt Template ─────────────────────────────────────────────────
    private fun buildPhi3Prompt(userMessage: String): String =
        "<|system|>\n$SYSTEM_PROMPT<|end|>\n" +
        conversationHistory.toString() +
        "<|user|>\n$userMessage<|end|>\n" +
        "<|assistant|>\n"

    // ── Asset Helper ──────────────────────────────────────────────────────────
    private fun copyAssetToFilesDir(assetName: String): String {
        val file = File(filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            Log.d(TAG, "$assetName already in filesDir — skipping copy")
            return file.absolutePath
        }

        Log.d(TAG, "Copying $assetName from assets…")
        assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                val buffer = ByteArray(8 * 1024)
                var n: Int
                while (input.read(buffer).also { n = it } != -1)
                    output.write(buffer, 0, n)
                output.flush()
            }
        }
        Log.d(TAG, "✅ Copied $assetName (${file.length() / 1024 / 1024}MB)")
        return file.absolutePath
    }

    // ── Utilities ─────────────────────────────────────────────────────────────
    private fun checkAvailableMemoryMB(): Long {
        val rt = Runtime.getRuntime()
        return (rt.maxMemory() - rt.totalMemory() + rt.freeMemory()) / 1024 / 1024
    }

    private fun scrollToBottom() {
        if (chatAdapter.itemCount > 0)
            recyclerView.scrollToPosition(chatAdapter.itemCount - 1)
    }

    private fun hideKeyboard() {
        val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
        imm.hideSoftInputFromWindow(currentFocus?.windowToken, 0)
    }
}
