try:
    from IPython.terminal.prompts import Prompts, Token
    from IPython.terminal.embed import embed as std_embed

    class CustomPrompt(Prompts):
        name = ""

        def __init__(self, shell):
            Prompts.__init__(self, shell)

        def setName(self, name):
            self.name = name

        def in_prompt_tokens(self, cli=None):
            return [
                    (Token.Prompt, self.name),
                    (Token.Prompt, ' In ['),
                    (Token.PromptNum, str(self.shell.execution_count)),
                    (Token.Prompt, ']: '),
            ]

        def out_prompt_tokens(self):
            return [
                    (Token.OutPrompt, self.name),
                    (Token.OutPrompt, ' Out['),
                    (Token.OutPromptNum, str(self.shell.execution_count)),
                    (Token.OutPrompt, ']: '),
            ]

    def embed(name: str="", loc_vars: dict=None):
        from traitlets.config import Config

        glob_vars = globals()
        if loc_vars is None:
            glob_vars.update(locals())
        else:
            glob_vars.update(loc_vars)

        cfg = Config()
        cfg.InteractiveShellApp.gui = "qt5"
        cfg.TerminalInteractiveShell.prompts_class = CustomPrompt
        CustomPrompt.name = name
        std_embed(config=cfg,
                  banner1='Dropping into IPython, type %gui qt5 to unlock GUI')

except ImportError:
    import code

    def embed(name: str="", loc_vars: dict=None):
        import sys
        sys.ps1 = name + " >>> "
        sys.ps2 = name + " ... "

        glob_vars = globals()
        if loc_vars is None:
            glob_vars.update(locals())
        else:
            glob_vars.update(loc_vars)
        shell = code.InteractiveConsole(glob_vars)
        shell.interact()
