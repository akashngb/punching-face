// Tool registry + dispatcher (OMNI.md §3.4).
//
// Scenarios register a set of tools once. The model then calls them via
// function-calling; the dispatcher validates arguments against the schema and
// forwards to the handler. Handlers are pure w.r.t. the model — they can
// mutate the rig, blendshapes, UI, but they never call the model back
// directly. If they need to, they emit an EventBus event that events.js will
// forward as engine context.
//
// The schemas conform to OpenAI-style function tools; the same shape is
// accepted by Alibaba's Realtime `session.update`'s `tools` field.

export class ToolRegistry {
  constructor({bus, session, onError} = {}) {
    this.bus = bus;
    this.session = session;
    this.onError = onError || ((error, tool) => console.warn('[omni-tool]', tool, error));
    this.tools = new Map();
  }

  /**
   * Register one tool.
   * @param {string} name
   * @param {{description:string, parameters:object, handler:(args, ctx)=>any}} tool
   */
  register(name, tool) {
    if (!name || typeof tool?.handler !== 'function') throw new Error('tool needs handler');
    this.tools.set(name, {name, ...tool});
    return this;
  }

  /** All schemas, formatted as `{type:'function', function:{...}}`. */
  schemas() {
    return [...this.tools.values()].map(t => ({
      type: 'function',
      name: t.name,             // Realtime format uses top-level name
      function: {name: t.name, description: t.description, parameters: t.parameters},
      description: t.description,
      parameters: t.parameters,
    }));
  }

  names() { return [...this.tools.keys()]; }

  /** Called with the tool-call event from OmniSession. Returns the handler's
   * return value (may be a promise). The dispatcher also emits a bus event
   * `tool.applied` so scenarios (and tests) can react without hooking on the
   * session directly. */
  async dispatch({name, arguments: rawArgs, call_id} = {}, ctx = {}) {
    const tool = this.tools.get(name);
    if (!tool) {
      this.onError(new Error(`unknown tool: ${name}`), name);
      return;
    }
    const args = validate(tool.parameters, rawArgs);
    if (args._error) {
      this.onError(new Error(args._error), name);
      return;
    }
    try {
      const result = await tool.handler(args, ctx);
      this.bus?.emit({type: 'tool.applied', tool: name, args, result: result ?? null});
      // Return the result to the model when we have a call_id — that closes
      // the OpenAI-style function loop so the model can continue reasoning.
      if (call_id && this.session) {
        this.session.socket?.send(JSON.stringify({
          type: 'conversation.item.create',
          item: {type: 'function_call_output', call_id, output: JSON.stringify(result ?? {ok: true})}
        }));
      }
      return result;
    } catch (error) {
      this.onError(error, name);
      this.bus?.emit({type: 'tool.failed', tool: name, args, error: String(error?.message || error)});
    }
  }
}

/**
 * Trivial validator: checks required keys and enum values. Coerces missing
 * booleans to `false` and missing numbers to null. Rich validation belongs in
 * the handler; we keep this small so a model mis-fire doesn't crash the UI.
 */
export function validate(schema, argsIn) {
  const args = argsIn && typeof argsIn === 'object' ? {...argsIn} : {};
  if (!schema || typeof schema !== 'object') return args;
  const {properties = {}, required = []} = schema;
  for (const key of required) {
    if (args[key] === undefined || args[key] === null || args[key] === '') {
      return {_error: `missing required arg: ${key}`};
    }
  }
  for (const [key, spec] of Object.entries(properties)) {
    if (args[key] == null) continue;
    if (spec.enum && !spec.enum.includes(args[key])) {
      return {_error: `${key}=${args[key]} not in ${spec.enum.join('|')}`};
    }
    if (spec.type === 'number' && typeof args[key] !== 'number') {
      const num = Number(args[key]);
      if (Number.isFinite(num)) args[key] = num;
      else return {_error: `${key} not numeric`};
    }
    if (spec.type === 'boolean') args[key] = !!args[key];
  }
  return args;
}
