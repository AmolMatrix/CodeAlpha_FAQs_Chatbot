const chatBox = document.getElementById("chat-box");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const typingIndicator = document.getElementById("typing-indicator");
const tabs = document.querySelectorAll(".tab");
const chipGroups = document.querySelectorAll(".chip-group");
const clearBtn = document.getElementById("clear-btn");

// ---- Category tab filtering ----
function showCategory(cat) {
  tabs.forEach((t) => t.classList.toggle("active", t.dataset.cat === cat));
  chipGroups.forEach((g) => {
    g.classList.toggle("visible", cat === "all" || g.dataset.group === cat);
  });
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => showCategory(tab.dataset.cat));
});

showCategory("all");

// ---- Chat message rendering ----
function addMessage(text, sender, meta) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${sender}`;

  if (sender === "bot") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "🤖";
    wrapper.appendChild(avatar);
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  if (sender === "bot" && meta && meta.confidence !== undefined) {
    const barWrap = document.createElement("div");
    barWrap.className = "confidence-bar";
    const fill = document.createElement("div");
    fill.className = "confidence-fill";
    fill.style.width = "0%";
    barWrap.appendChild(fill);
    bubble.appendChild(barWrap);

    const label = document.createElement("span");
    label.className = "confidence-label";
    label.textContent = `Confidence: ${meta.confidence}%`;
    bubble.appendChild(label);

    wrapper.appendChild(bubble);
    chatBox.appendChild(wrapper);
    // animate the bar after it's in the DOM
    requestAnimationFrame(() => {
      fill.style.width = `${Math.min(meta.confidence, 100)}%`;
    });
  } else {
    wrapper.appendChild(bubble);
    chatBox.appendChild(wrapper);
  }

  chatBox.scrollTop = chatBox.scrollHeight;
}

function showTyping(show) {
  typingIndicator.classList.toggle("active", show);
  if (show) chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage(message) {
  addMessage(message, "user");
  userInput.value = "";
  showTyping(true);

  try {
    // small delay so the typing indicator feels natural
    const [response] = await Promise.all([
      fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      }),
      new Promise((r) => setTimeout(r, 400)),
    ]);
    const data = await response.json();

    showTyping(false);

    if (data.error) {
      addMessage(data.error, "bot");
      return;
    }
    addMessage(data.answer, "bot", { confidence: data.confidence });
  } catch (err) {
    showTyping(false);
    addMessage("Something went wrong. Please make sure the server is running.", "bot");
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = userInput.value.trim();
  if (message) sendMessage(message);
});

function askSuggested(button) {
  sendMessage(button.textContent);
}

// ---- Clear chat ----
clearBtn.addEventListener("click", () => {
  chatBox.innerHTML = "";
  addMessage("Chat cleared. Ask me anything about SGMCOE!", "bot");
});
