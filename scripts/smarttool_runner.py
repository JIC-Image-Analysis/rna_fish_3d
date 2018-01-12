from smarttoolbase import SmartTool, Command, parse_args

BASE_COMMANDS = [
    Command("python " + "scripts/analysis.py" + " {input_fpath} .")]
OUTPUTS = [
    "enhanced_annotated_channel_0.png",
    "enhanced_annotated_channel_1.png",
    "dapi_channel_2.png",
]

class RnaFish3DTool(SmartTool):

    def pre_run(self, identifier):
        input_fpath = self.input_dataset.item_content_abspath(identifier)

        self.base_command_props.update(
            {'input_fpath': input_fpath}
        )


def main():
    args = parse_args()

    with RnaFish3DTool(args.input_uri, args.output_uri) as smart_tool:
        smart_tool.base_commands = BASE_COMMANDS
        smart_tool.outputs = OUTPUTS
        smart_tool(args.identifier)


if __name__ == '__main__':
    main()
