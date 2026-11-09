type Reply = {id: string; status: number; data?: unknown};

let active: {id: string; finish: (reply: Reply) => void} | undefined;
let queue: Promise<unknown> = Promise.resolve();
let parentOrigin = '*';

function post(type: string, payload: object) {
  window.parent.postMessage({isStreamlitMessage: true, type, ...payload}, parentOrigin);
}

export function connectStreamlit(root: HTMLElement) {
  window.addEventListener('message', (event) => {
    if (event.source !== window.parent || event.data?.type !== 'streamlit:render') return;
    if (parentOrigin !== '*' && parentOrigin !== event.origin) return;
    parentOrigin = event.origin;
    const reply: Reply | undefined = event.data.args?.reply;
    if (reply && active?.id === reply.id) active.finish(reply);
  });

  let frameHeight = 0;
  new ResizeObserver(() => {
    const height = Math.ceil(root.getBoundingClientRect().height);
    if (height > 0 && height !== frameHeight) {
      frameHeight = height;
      post('streamlit:setFrameHeight', {height});
    }
  }).observe(root);

  post('streamlit:componentReady', {apiVersion: 1});
}

function send(path: string, options?: RequestInit): Promise<Reply> {
  const signal = options?.signal;
  signal?.throwIfAborted();

  return new Promise((resolve, reject) => {
    const id = crypto.randomUUID();
    const timeout = window.setTimeout(() => {
      active = undefined;
      reject(new Error('The Streamlit model did not respond'));
    }, 60_000);
    active = {
      id,
      finish(reply) {
        window.clearTimeout(timeout);
        active = undefined;
        resolve(reply);
      },
    };
    post('streamlit:setComponentValue', {
      dataType: 'json',
      value: {id, path, body: options?.body ? JSON.parse(String(options.body)) : null},
    });
  });
}

export function streamlitRequest(path: string, options?: RequestInit): Promise<Reply> {
  // Streamlit keeps the latest widget value. Send one request at a time so a
  // newer search cannot replace an in-flight recommendation or profile request.
  const request = queue.then(() => send(path, options));
  queue = request.catch(() => undefined);
  return request.then((reply) => {
    options?.signal?.throwIfAborted();
    return reply;
  });
}
