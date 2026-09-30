(function () {
  const panel = document.getElementById("chat-panel");
  const log = document.getElementById("chat-log");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send");
  const suggest = document.getElementById("chat-suggest");
  let busy = false;

  function openChat() {
    panel.hidden = false;
    input.focus();
  }
  function closeChat() {
    panel.hidden = true;
  }

  document.querySelectorAll("[data-open-chat]").forEach((el) =>
    el.addEventListener("click", openChat)
  );
  document.querySelectorAll("[data-close-chat]").forEach((el) =>
    el.addEventListener("click", closeChat)
  );
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !panel.hidden) closeChat();
  });

  // Any element with data-ask="..." opens the chat and sends that question.
  document.addEventListener("click", (e) => {
    const trigger = e.target.closest("[data-ask]");
    if (trigger) {
      openChat();
      ask(trigger.dataset.ask);
    }
  });

  function addMessage(who, text, extra) {
    const wrap = document.createElement("div");
    wrap.className = "msg " + who;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (extra && extra.abstained) bubble.classList.add("abstain");
    bubble.textContent = text;
    wrap.appendChild(bubble);

    if (extra && typeof extra.confidence === "number") {
      const meta = document.createElement("div");
      meta.className = "meta";
      const pct = Math.round(extra.confidence * 100);
      const bar = document.createElement("span");
      bar.className = "conf-bar";
      const fill = document.createElement("span");
      fill.style.width = Math.max(pct, 3) + "%";
      bar.appendChild(fill);
      const label = document.createElement("span");
      label.textContent = (extra.abstained ? "Not sure" : "Confidence " + pct + "%");
      meta.appendChild(bar);
      meta.appendChild(label);
      wrap.appendChild(meta);
      if (extra.matched) {
        const m = document.createElement("div");
        m.className = "matched";
        m.textContent = "Closest known question: " + extra.matched;
        wrap.appendChild(m);
      }
    }
    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;
    return wrap;
  }

  async function ask(question) {
    question = (question || "").trim();
    if (!question || busy) return;
    busy = true;
    sendBtn.disabled = true;
    suggest.hidden = true;
    addMessage("user", question);
    const typing = addMessage("bot", "Thinking...");
    typing.classList.add("typing");

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      typing.remove();
      if (!res.ok) {
        addMessage("bot", data.error || "Something went wrong. Please try again.");
      } else {
        addMessage("bot", data.answer, {
          confidence: data.confidence,
          abstained: data.abstained,
          matched: data.matched_question,
        });
      }
    } catch (err) {
      typing.remove();
      addMessage("bot", "I could not reach the server. Is the app still running?");
    } finally {
      busy = false;
      sendBtn.disabled = false;
      input.focus();
    }
  }

  window.askVIT = function (q) {
    openChat();
    ask(q);
  };

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value;
    input.value = "";
    ask(q);
  });
})();
