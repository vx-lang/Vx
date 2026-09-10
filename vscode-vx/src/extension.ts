import * as path from 'path';
import { workspace, ExtensionContext } from 'vscode';

import {
	LanguageClient,
	LanguageClientOptions,
	ServerOptions,
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: ExtensionContext) {
	// The vx-analyzer binary: an explicit setting if there is one, otherwise whatever is
	// on PATH. This was an absolute path to one machine's debug build, so the extension
	// could never have started the server anywhere else.
	const configured = workspace.getConfiguration('vx').get<string>('analyzerPath');
	const serverCommand = configured && configured.trim().length > 0
		? configured
		: 'vx-analyzer';

	const serverOptions: ServerOptions = {
		run: { command: serverCommand },
		debug: { command: serverCommand }
	};

	const clientOptions: LanguageClientOptions = {
		documentSelector: [{ scheme: 'file', language: 'vx' }],
		synchronize: {
			fileEvents: workspace.createFileSystemWatcher('**/*.vx')
		}
	};

	client = new LanguageClient(
		'vxLanguageServer',
		'Vx Language Server',
		serverOptions,
		clientOptions
	);

	client.start().catch((err: any) => {
		import('vscode').then(vscode => {
			vscode.window.showErrorMessage(`Failed to start Vx Language Server: ${err.message}`);
		});
	});
}

export function deactivate(): Thenable<void> | undefined {
	if (!client) {
		return undefined;
	}
	return client.stop();
}
