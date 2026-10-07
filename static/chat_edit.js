function setupChatEditing(options) {
    const chat = options.chat;
    const getWs = options.getWs;
    const prompts = options.prompts || {};
    const onHistoryChange = options.onHistoryChange || function () {};
    let turnBusy = false;
    let streamingContentEl = null;

    function setBusy(value) {
        turnBusy = !!value;
        document.querySelectorAll('.message-actions button').forEach((btn) => {
            btn.disabled = turnBusy;
        });
        if (options.onBusy) options.onBusy(turnBusy);
    }

    function actionButton(label, title, handler) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = label;
        btn.title = title;
        btn.addEventListener('click', function (e) {
            e.preventDefault();
            e.stopPropagation();
            if (turnBusy) return;
            handler();
        });
        return btn;
    }

    function addMessageActions(messageDiv, role) {
        if (role !== 'user' && role !== 'assistant') return;
        if (!messageDiv.dataset.id) return;
        if (messageDiv.querySelector('.message-actions')) return;
        const actions = document.createElement('div');
        actions.className = 'message-actions';
        if (role === 'user') {
            actions.appendChild(actionButton('правка', 'Переписать сообщение', function () {
                startRewrite(messageDiv);
            }));
            actions.appendChild(actionButton('удалить', 'Удалить это и все сообщения после него', function () {
                sendAction('delete', messageDiv.dataset.id);
            }));
        } else {
            actions.appendChild(actionButton('заново', 'Перегенерировать ответ модели', function () {
                sendAction('regenerate', messageDiv.dataset.id);
            }));
            actions.appendChild(actionButton('удалить', 'Удалить это и все сообщения после него', function () {
                sendAction('delete', messageDiv.dataset.id);
            }));
        }
        messageDiv.appendChild(actions);
        if (turnBusy) {
            actions.querySelectorAll('button').forEach((btn) => { btn.disabled = true; });
        }
    }

    function sendAction(type, id, extra) {
        const ws = getWs();
        if (!ws || ws.readyState !== WebSocket.OPEN || !id) return;
        setBusy(true);
        const payload = Object.assign({ type: type, id: id }, extra || {});
        ws.send(JSON.stringify(payload));
    }

    function startRewrite(messageDiv) {
        if (turnBusy || messageDiv.classList.contains('editing')) return;
        const contentSpan = messageDiv.querySelector('.message-content');
        const actions = messageDiv.querySelector('.message-actions');
        if (!contentSpan) return;
        const original = contentSpan.textContent;
        messageDiv.classList.add('editing');
        const ta = document.createElement('textarea');
        ta.className = 'rewrite-input';
        ta.value = original;
        contentSpan.style.display = 'none';
        contentSpan.insertAdjacentElement('afterend', ta);
        ta.focus();
        ta.setSelectionRange(ta.value.length, ta.value.length);

        const editActions = document.createElement('div');
        editActions.className = 'message-actions';
        const saveBtn = actionButton('сохранить', 'Сохранить и получить новый ответ', function () {
            const text = ta.value.trim();
            if (!text) return;
            sendAction('rewrite', messageDiv.dataset.id, { content: text });
        });
        const cancelBtn = actionButton('отмена', 'Отменить правку', function () {
            ta.remove();
            editActions.remove();
            contentSpan.style.display = '';
            messageDiv.classList.remove('editing');
            if (actions) actions.style.display = '';
        });
        editActions.appendChild(saveBtn);
        editActions.appendChild(cancelBtn);
        if (actions) actions.style.display = 'none';
        messageDiv.appendChild(editActions);
    }

    function truncateFromId(id) {
        if (!id) return;
        const rows = Array.from(chat.querySelectorAll('.message'));
        const start = rows.findIndex((row) => row.dataset.id === id);
        if (start < 0) return;
        for (let i = start; i < rows.length; i++) {
            rows[i].remove();
        }
        streamingContentEl = null;
        onHistoryChange();
    }

    function addMessage(role, content, id) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ' + role;
        if (id) messageDiv.dataset.id = id;

        const prompt = document.createElement('span');
        prompt.className = 'message-prompt';
        if (role === 'user') {
            prompt.textContent = prompts.user || 'user@system:~$ ';
        } else if (role === 'assistant') {
            prompt.textContent = prompts.assistant || 'coreline@system:~$ ';
        } else {
            prompt.textContent = prompts.system || '[SYSTEM] ';
        }

        const contentSpan = document.createElement('span');
        contentSpan.className = 'message-content';
        contentSpan.textContent = content;

        messageDiv.appendChild(prompt);
        messageDiv.appendChild(contentSpan);
        chat.appendChild(messageDiv);
        addMessageActions(messageDiv, role);
        chat.scrollTop = chat.scrollHeight;
        onHistoryChange();
        return contentSpan;
    }

    function startStreamingMessage(id) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';
        if (id) messageDiv.dataset.id = id;
        const prompt = document.createElement('span');
        prompt.className = 'message-prompt';
        prompt.textContent = prompts.assistant || 'coreline@system:~$ ';
        const contentSpan = document.createElement('span');
        contentSpan.className = 'message-content';
        messageDiv.appendChild(prompt);
        messageDiv.appendChild(contentSpan);
        chat.appendChild(messageDiv);
        streamingContentEl = contentSpan;
        chat.scrollTop = chat.scrollHeight;
    }

    function appendStreamingChunk(content) {
        if (streamingContentEl) {
            streamingContentEl.textContent += (content || '').replace(/\[TIME\]/gi, '');
            chat.scrollTop = chat.scrollHeight;
        }
    }

    function endStreamingMessage(finalContent, id) {
        if (streamingContentEl) {
            if (finalContent !== undefined && finalContent !== null) {
                streamingContentEl.textContent = finalContent;
            }
            const parent = streamingContentEl.parentElement;
            if (parent && id) parent.dataset.id = id;
            if (parent) addMessageActions(parent, 'assistant');
        }
        streamingContentEl = null;
        chat.scrollTop = chat.scrollHeight;
        onHistoryChange();
    }

    function handleServerEvent(data) {
        if (data.type === 'message_start') {
            startStreamingMessage((data.data && data.data.id) || '');
            return true;
        }
        if (data.type === 'message_chunk') {
            appendStreamingChunk((data.data && data.data.content) || '');
            return true;
        }
        if (data.type === 'truncate') {
            truncateFromId(data.data && data.data.id);
            return true;
        }
        if (data.type === 'turn_done') {
            setBusy(false);
            return true;
        }
        if (data.type === 'message') {
            const role = data.data.role;
            const content = data.data.content;
            const id = data.data.id;
            if (streamingContentEl && role === 'assistant') {
                endStreamingMessage(content, id);
            } else {
                addMessage(role, content, id);
            }
            return true;
        }
        return false;
    }

    return {
        addMessage,
        setBusy,
        handleServerEvent,
        isBusy: function () { return turnBusy; },
        hasStreaming: function () { return !!streamingContentEl; }
    };
}
